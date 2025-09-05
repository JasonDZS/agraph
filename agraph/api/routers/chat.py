"""Chat functionality router."""

from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ...logger import logger
from ..dependencies import get_agraph_instance
from ..models import ChatRequest, ChatResponse, ResponseStatus, StreamChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific chat"),
) -> ChatResponse:
    """Chat with the knowledge base (non-streaming)."""
    try:
        # Get the appropriate AGraph instance based on project_name
        agraph = await get_agraph_instance(project_name)
        if request.stream:
            raise HTTPException(status_code=400, detail="Use /chat/stream endpoint for streaming responses")

        response = await agraph.chat(
            question=request.question,
            conversation_history=request.conversation_history,
            entity_top_k=request.entity_top_k,
            relation_top_k=request.relation_top_k,
            text_chunk_top_k=request.text_chunk_top_k,
            response_type=request.response_type,
            stream=False,
        )

        # Ensure response is a dict for non-streaming
        if not isinstance(response, dict):
            raise HTTPException(status_code=500, detail="Invalid response format from chat engine")

        return ChatResponse(
            status=ResponseStatus.SUCCESS,
            message="Chat response generated successfully",
            data=response,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        # Note: Don't close the instance here as it's cached and reused
        pass


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    project_name: Optional[str] = Query(default=None, description="Project name for project-specific chat"),
) -> StreamingResponse:
    """Chat with the knowledge base (streaming)."""
    try:
        # Get the appropriate AGraph instance based on project_name
        agraph = await get_agraph_instance(project_name)

        async def generate() -> AsyncGenerator[str, None]:
            try:
                chat_response = await agraph.chat(
                    question=request.question,
                    conversation_history=request.conversation_history,
                    entity_top_k=request.entity_top_k,
                    relation_top_k=request.relation_top_k,
                    text_chunk_top_k=request.text_chunk_top_k,
                    response_type=request.response_type,
                    stream=True,
                )

                # Handle async generator response for streaming
                if hasattr(chat_response, "__aiter__"):
                    async for chunk_data in chat_response:
                        response = StreamChatResponse(**chunk_data)
                        yield f"data: {response.model_dump_json()}\n\n"
                else:
                    # Handle non-streaming response as single chunk
                    response = StreamChatResponse(**chat_response)
                    yield f"data: {response.model_dump_json()}\n\n"
            except Exception as e:
                error_response = StreamChatResponse(
                    question=request.question,
                    chunk="",
                    partial_answer=f"Error: {str(e)}",
                    finished=True,
                )
                yield f"data: {error_response.model_dump_json()}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
