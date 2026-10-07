from __future__ import annotations

import importlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

try:  # pragma: no cover - runtime dependency check
    _rapidfuzz_module = importlib.import_module("rapidfuzz.fuzz")
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "rapidfuzz is required but not installed. Run `pip install rapidfuzz`."
    ) from exc

token_sort_ratio = _rapidfuzz_module.token_sort_ratio

from server.core.clip_utils import embed_image, embed_text, cosine_np
from server.core.cv_detect import detect_and_crop_slide
from server.core.gemini import (
    GeminiError,
    answer_question_with_context,
    summarize_slide,
)
from server.core.image_utils import decode_image
from server.core.text_ocr import extract_text
from server.models.summary_schema import (
    BoundingBoxPoint,
    CompareSlidesResponse,
    GeminiSummary,
    ProcessSlideResponse,
    SlideComparisonMetrics,
    SlideVisualDetails,
)
from server.storage import (
    append_slide_history,
    ensure_session,
    load_last_state,
    load_slide_history,
    log_qa_turn,
    save_last_state,
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["slides"])


TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.60"))
CLIP_THRESHOLD = float(os.getenv("CLIP_THRESHOLD", "0.88"))
USE_BBOX_FOR_ANALYSIS = os.getenv("USE_BBOX", "0") == "1"
QUESTION_RELEVANCE_THRESHOLD = float(
    os.getenv("QUESTION_RELEVANCE_THRESHOLD", "0.20")
)
MAX_SLIDE_CONTEXT = int(os.getenv("SLIDE_CONTEXT_LIMIT", "5"))
MAX_CONVERSATION_MEMORY = int(os.getenv("CONVERSATION_MEMORY_LIMIT", "5"))
RETRIEVAL_CONTEXT_LIMIT = int(os.getenv("RETRIEVAL_CONTEXT_LIMIT", "3"))
RECENT_CONTEXT_FALLBACK = int(os.getenv("RECENT_CONTEXT_FALLBACK", "2"))

# Extremely small sets to keep obvious off-topic queries from being answered.
# Anything not matching these will be treated as related enough to proceed.
OFF_TOPIC_KEYWORDS = {
    "soccer",
    "football",
    "basketball",
    "baseball",
    "harry potter",
    "taylor swift",
    "celebrity",
    "movie",
    "music video",
    "recipe",
    "cook",
    "pizza",
    "burger",
    "weather",
    "vacation",
    "travel",
}

ACADEMIC_HINTS = {
    "activation",
    "algorithm",
    "analysis",
    "architecture",
    "classification",
    "function",
    "gradient",
    "graph",
    "lecture",
    "model",
    "neural",
    "network",
    "optimization",
    "probability",
    "regression",
    "statistics",
    "summary",
    "training",
}


_previous_text_by_session: dict[str, Optional[str]] = {}
_previous_clip_vec_by_session: dict[str, np.ndarray] = {}
_initialized_sessions: set[str] = set()
_conversation_memory_by_session: dict[str, List[dict[str, object]]] = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    slide_summary: Optional[Dict[str, Any]] = None


def _corners_to_points(corners: Optional[np.ndarray]) -> Optional[List[BoundingBoxPoint]]:
    if corners is None:
        return None
    return [BoundingBoxPoint(x=int(point[0]), y=int(point[1])) for point in corners]


def _ensure_previous_state_loaded(session_id: str) -> None:
    if session_id in _initialized_sessions:
        return
    state = load_last_state(session_id=session_id)
    _previous_text_by_session[session_id] = state.text
    if state.clip_vector is not None:
        _previous_clip_vec_by_session[session_id] = state.clip_vector.astype(np.float32)
    _initialized_sessions.add(session_id)


def _record_conversation(
    session_id: str,
    question: str,
    answer: str,
    slide_number: int,
) -> None:
    memory = _conversation_memory_by_session.setdefault(session_id, [])
    memory.append(
        {
            "question": question.strip(),
            "answer": answer.strip(),
            "slide_number": slide_number,
        }
    )
    if len(memory) > MAX_CONVERSATION_MEMORY:
        del memory[:-MAX_CONVERSATION_MEMORY]


def _build_conversation_text(session_id: str) -> str:
    memory = _conversation_memory_by_session.get(session_id, [])
    if not memory:
        return ""
    segments = []
    for turn in memory[-MAX_CONVERSATION_MEMORY:]:
        segments.append(
            f"Slide {turn['slide_number']} - User: {turn['question']}\nAssistant: {turn['answer']}"
        )
    return "\n\n".join(segments)


def _topic_hint(summary: Optional[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    titles = summary.get("title") if isinstance(summary, dict) else None
    if titles and isinstance(titles, list) and titles:
        return titles[0]
    summary_list = summary.get("summary") if isinstance(summary, dict) else None
    if summary_list and isinstance(summary_list, list) and summary_list:
        return summary_list[0][:80]
    return ""


def _slide_text(entry: Dict[str, Any]) -> str:
    summary = entry.get("summary", {}) or {}
    parts = [json.dumps(summary, ensure_ascii=False)]
    ocr_text = entry.get("ocr_text")
    if ocr_text:
        parts.append(str(ocr_text))
    return "\n".join(parts)


def _build_slide_context(entries: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
    if not entries:
        return ""
    selected = entries[-limit:] if limit is not None and limit > 0 else entries
    parts: List[str] = []
    for entry in selected:
        summary_blob = json.dumps(entry.get("summary", {}), ensure_ascii=False)
        ocr_text = entry.get("ocr_text")
        if ocr_text:
            summary_blob = f"{summary_blob}\nOCR text: {ocr_text}"
        parts.append(
            f"Slide {entry.get('slide_number', '?')} (captured at {entry.get('timestamp', 'unknown')}):\n{summary_blob}"
        )
    return "\n\n".join(parts)


def _score_slide_for_question(question: str, entry: Dict[str, Any]) -> float:
    slide_text = _slide_text(entry)
    if not slide_text.strip():
        return 0.0

    question_tokens = _question_tokens(question)
    slide_tokens = _question_tokens(slide_text)
    if not question_tokens or not slide_tokens:
        return 0.0

    overlap = len(question_tokens & slide_tokens) / max(len(question_tokens), 1)
    fuzzy_score = token_sort_ratio(question, slide_text[:2500]) / 100.0
    return (0.70 * overlap) + (0.30 * fuzzy_score)


def _select_relevant_slide_context(
    question: str,
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not history:
        return []

    scored = [
        (_score_slide_for_question(question, entry), index, entry)
        for index, entry in enumerate(history)
    ]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: List[Dict[str, Any]] = []
    seen_numbers: set[int] = set()

    for score, _, entry in scored[:RETRIEVAL_CONTEXT_LIMIT]:
        if score <= 0:
            continue
        slide_number = entry.get("slide_number")
        if isinstance(slide_number, int):
            seen_numbers.add(slide_number)
        selected.append(entry)

    for entry in history[-RECENT_CONTEXT_FALLBACK:]:
        slide_number = entry.get("slide_number")
        if isinstance(slide_number, int) and slide_number in seen_numbers:
            continue
        selected.append(entry)
        if isinstance(slide_number, int):
            seen_numbers.add(slide_number)

    if not selected:
        selected = history[-MAX_SLIDE_CONTEXT:]

    return sorted(
        selected,
        key=lambda entry: entry.get("slide_number", 0),
    )[-MAX_SLIDE_CONTEXT:]


def _append_slide_history_entry(
    summary: Dict[str, Any],
    *,
    session_id: str,
    ocr_text: str,
    text_similarity: Optional[float],
    clip_cosine: Optional[float],
) -> Dict[str, Any]:
    history = load_slide_history(session_id=session_id)
    slide_number = history[-1]["slide_number"] + 1 if history else 1
    entry = {
        "slide_number": slide_number,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary,
        "ocr_text": ocr_text,
        "metrics": {
            "text_similarity": text_similarity,
            "clip_cosine": clip_cosine,
        },
    }
    append_slide_history(entry, session_id=session_id)
    return entry


def _question_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\b[a-zA-Z][\w-]*\b", text.lower()) if token}


def _is_question_obviously_unrelated(question: str, summary: Dict[str, Any]) -> bool:
    """Return True only when the question is clearly unrelated to the slide."""
    question_tokens = _question_tokens(question)
    if not question_tokens:
        return False

    summary_text = json.dumps(summary or {}, ensure_ascii=False).lower()
    summary_tokens = _question_tokens(summary_text)

    # If we see any shared tokens or obvious academic hints, treat as related.
    if question_tokens & summary_tokens:
        return False
    if any(hint in question_tokens for hint in ACADEMIC_HINTS):
        return False

    # Otherwise only reject if we detect an explicit off-topic keyword.
    question_text = question.lower()
    return any(keyword in question_text for keyword in OFF_TOPIC_KEYWORDS)


@dataclass
class _SlideSample:
    analysis_image: np.ndarray
    clip_vector: np.ndarray
    ocr_text: str
    visual: SlideVisualDetails


def _process_frame(frame: np.ndarray) -> _SlideSample:
    cropped, detected, corners = detect_and_crop_slide(frame)
    analysis_region = cropped if detected and USE_BBOX_FOR_ANALYSIS else frame

    clip_tensor = embed_image(analysis_region)
    clip_vector = clip_tensor.detach().cpu().numpy()[0].astype(np.float32)

    ocr_text = extract_text(analysis_region)

    visual = SlideVisualDetails(
        slide_detected=detected,
        bounding_box=_corners_to_points(corners),
        cropped_image_base64=None,  # Bounding box previews disabled for now
        annotated_image_base64=None,
    )

    return _SlideSample(
        analysis_image=analysis_region,
        clip_vector=clip_vector,
        ocr_text=ocr_text,
        visual=visual,
    )


def _determine_change(
    current_text: str,
    current_clip: np.ndarray,
    *,
    session_id: str,
) -> tuple[bool, Optional[float], Optional[float]]:
    _ensure_previous_state_loaded(session_id)

    previous_text = _previous_text_by_session.get(session_id)
    previous_clip = _previous_clip_vec_by_session.get(session_id)
    if previous_text is None or previous_clip is None:
        return True, None, None

    text_similarity = token_sort_ratio(current_text, previous_text) / 100.0
    clip_cosine = cosine_np(current_clip, previous_clip)
    is_new = text_similarity < TEXT_THRESHOLD and clip_cosine < CLIP_THRESHOLD
    return is_new, text_similarity, clip_cosine


def _update_memory(
    *,
    session_id: str,
    text: str,
    clip_vector: np.ndarray,
) -> None:
    _previous_text_by_session[session_id] = text
    _previous_clip_vec_by_session[session_id] = clip_vector


@router.post(
    "/process_slide",
    response_model=ProcessSlideResponse,
    summary="Process a slide image using CLIP and OCR similarity heuristics.",
)
async def process_slide(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> ProcessSlideResponse:
    current_session_id = ensure_session(session_id)
    try:
        raw_bytes = await image.read()
        frame = decode_image(raw_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {exc}",
        ) from exc

    sample = _process_frame(frame)
    is_new, text_similarity, clip_cosine = _determine_change(
        sample.ocr_text,
        sample.clip_vector,
        session_id=current_session_id,
    )

    summary_payload = None
    current_slide_number: Optional[int] = None

    if is_new:
        logger.info(
            "New slide detected for session %s (text=%.3f, clip=%.3f); invoking Gemini.",
            current_session_id,
            text_similarity if text_similarity is not None else -1.0,
            clip_cosine if clip_cosine is not None else -1.0,
        )
        try:
            summary_payload = summarize_slide(sample.analysis_image, ocr_text=sample.ocr_text)
        except GeminiError as exc:
            logger.error("Gemini summarization failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini summarization failed: {exc}",
            ) from exc

        save_last_state(
            sample.analysis_image,
            summary=summary_payload,
            text=sample.ocr_text,
            clip_vector=sample.clip_vector,
            session_id=current_session_id,
        )
        _update_memory(
            session_id=current_session_id,
            text=sample.ocr_text,
            clip_vector=sample.clip_vector,
        )

        if summary_payload is not None:
            history_entry = _append_slide_history_entry(
                summary_payload,
                session_id=current_session_id,
                ocr_text=sample.ocr_text,
                text_similarity=text_similarity,
                clip_cosine=clip_cosine,
            )
            current_slide_number = history_entry["slide_number"]
        else:
            history = load_slide_history(session_id=current_session_id)
            current_slide_number = (history[-1]["slide_number"] + 1) if history else 1
    else:
        logger.info(
            "Slide unchanged for session %s (text=%.3f, clip=%.3f); reusing cached summary.",
            current_session_id,
            text_similarity if text_similarity is not None else -1.0,
            clip_cosine if clip_cosine is not None else -1.0,
        )
        state = load_last_state(session_id=current_session_id)
        summary_payload = state.summary
        if summary_payload is None:
            logger.debug("No cached summary found; skipping summary payload.")
        save_last_state(
            sample.analysis_image,
            summary=summary_payload,
            text=sample.ocr_text,
            clip_vector=sample.clip_vector,
            session_id=current_session_id,
        )
        _update_memory(
            session_id=current_session_id,
            text=sample.ocr_text,
            clip_vector=sample.clip_vector,
        )
        history = load_slide_history(session_id=current_session_id)
        if history:
            current_slide_number = history[-1]["slide_number"]
        else:
            current_slide_number = 1

    summary_model = GeminiSummary.model_validate(summary_payload) if summary_payload else None

    return ProcessSlideResponse(
        new_slide=is_new,
        clip_cosine=clip_cosine if clip_cosine is not None else 0.0,
        text_similarity=text_similarity,
        slide_detected=sample.visual.slide_detected,
        bounding_box=sample.visual.bounding_box,
        summary=summary_model,
        slide_number=current_slide_number,
        session_id=current_session_id,
    )


@router.post(
    "/compare_slides",
    response_model=CompareSlidesResponse,
    summary="Compare two slide images using CLIP and OCR similarity heuristics.",
)
async def compare_slides(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
) -> CompareSlidesResponse:
    try:
        frame1 = decode_image(await image1.read())
        frame2 = decode_image(await image2.read())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {exc}",
        ) from exc

    sample1 = _process_frame(frame1)
    sample2 = _process_frame(frame2)

    text_similarity = token_sort_ratio(sample1.ocr_text, sample2.ocr_text) / 100.0
    clip_cosine = cosine_np(sample1.clip_vector, sample2.clip_vector)
    new_slide = text_similarity < TEXT_THRESHOLD and clip_cosine < CLIP_THRESHOLD

    metrics = SlideComparisonMetrics(
        clip_cosine=clip_cosine,
        text_similarity=text_similarity,
    )

    return CompareSlidesResponse(
        slide1=sample1.visual,
        slide2=sample2.visual,
        metrics=metrics,
        new_slide=new_slide,
    )


@router.post(
    "/ask",
    summary="Answer a question using the stored slide context.",
)
async def ask_question(payload: QuestionRequest) -> dict[str, object]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    current_session_id = ensure_session(payload.session_id)
    history = load_slide_history(session_id=current_session_id)
    if not history:
        return {
            "answer": "I don't have any slide context yet.",
            "session_id": current_session_id,
        }

    current_entry = history[-1]
    slide_number = current_entry.get("slide_number", len(history))
    summary_blob: Dict[str, Any] = (
        payload.slide_summary or current_entry.get("summary", {}) or {}
    )

    try:
        question_vec = embed_text(question)
        summary_vec = embed_text(json.dumps(summary_blob, ensure_ascii=False))
        relevance = cosine_np(question_vec, summary_vec)
    except Exception as exc:  # pragma: no cover - embedding failure
        logger.error("Failed to compute relevance: %s", exc)
        relevance = 0.0

    if relevance < QUESTION_RELEVANCE_THRESHOLD:
        if _is_question_obviously_unrelated(question, summary_blob):
            topic = _topic_hint(summary_blob)
            hint = f" ({topic})" if topic else ""
            return {
                "answer": f"This question does not appear related to the lecture slide topic{hint}.",
                "slide_number": slide_number,
                "relevance": relevance,
                "session_id": current_session_id,
            }
        else:
            logger.debug(
                "Treating low-relevance question as related (relevance=%.3f): %s",
                relevance,
                question,
            )

    selected_context = _select_relevant_slide_context(question, history)
    matched_slides = [
        entry.get("slide_number")
        for entry in selected_context
        if isinstance(entry.get("slide_number"), int)
    ]
    context_text = _build_slide_context(selected_context)
    conversation_text = _build_conversation_text(current_session_id)

    try:
        answer = answer_question_with_context(
            question=question,
            context=context_text,
            conversation=conversation_text,
        )
    except GeminiError as exc:
        logger.error("Gemini conversation response failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    _record_conversation(current_session_id, question, answer, slide_number)
    log_qa_turn(
        session_id=current_session_id,
        slide_number=slide_number if isinstance(slide_number, int) else None,
        question=question,
        answer=answer,
        relevance=relevance,
        matched_slides=matched_slides,
    )
    return {
        "answer": answer,
        "slide_number": slide_number,
        "relevance": relevance,
        "matched_slides": matched_slides,
        "session_id": current_session_id,
    }
