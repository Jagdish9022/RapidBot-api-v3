from fastapi import APIRouter, HTTPException
from app.db.models import QARequest
from app.utils.LangChain import add_to_conversation_history, get_simple_conversation_history
from app.services.gemini import ask_gemini_fast, enhanced_query_with_gemini, translate_to_english
from app.services.embeddings import get_question_embedding
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["question-answering"])

@router.post("/ask-question")
async def ask_question(req: QARequest):
    try:
        # Input validation
        if not req.question or req.question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if not req.collection_name or req.collection_name.strip() == "":
            raise HTTPException(status_code=400, detail="Collection name cannot be empty")

        logger.info(f"Processing question: '{req.question}' for collection: '{req.collection_name}'")
        
        # Get conversation history (super fast)
        conversation_history = get_simple_conversation_history(req.collection_name)
        logger.debug(f"Retrieved {len(conversation_history)} messages from history")

        # Translate query if needed
        try:
            translated_query = translate_to_english(req.question)
            logger.debug(f"Translated query: {translated_query}")
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")
            translated_query = req.question

        # Step 1: Get embedding
        try:
            question_embedding = get_question_embedding(translated_query)
            logger.debug(f"Generated embedding with {len(question_embedding)} dimensions")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise HTTPException(status_code=500, detail="Failed to process question")

        # Step 2: Enhanced query to get search results and context
        try:
            enhanced_results = enhanced_query_with_gemini(
                collection_name=req.collection_name,
                user_query=translated_query,
                query_vector=question_embedding,
                limit=10
            )
            logger.info(f"Enhanced query returned {enhanced_results.get('total_results', 0)} results")
        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            enhanced_results = {
                "error": str(e),
                "context_text": "Failed to retrieve relevant context.",
                "processed_query": {},
                "search_results": []
            }

        # Step 3: Ask Gemini with simple conversation history
        try:
            final_response = ask_gemini_fast(
                context=enhanced_results.get("context_text", ""),
                question=req.question,
                query_analysis=enhanced_results.get("processed_query", {}),
                enhanced_results=enhanced_results,
                conversation_history=conversation_history
            )
            logger.debug("Successfully generated Gemini response")
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            final_response = {
                "response": "I apologize, but I'm unable to process your request at the moment. Please try again later.",
                "buttons": False,
                "button_type": None,
                "button_data": None
            }

        # Save to conversation history (super fast)
        try:
            add_to_conversation_history(req.collection_name, "user", req.question)
            add_to_conversation_history(req.collection_name, "assistant", final_response["response"])
            logger.debug("Saved conversation to simple memory")
        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")

        # Add debug info to response
        final_response["conversation_id"] = req.collection_name
        final_response["debug_info"] = {
            "total_search_results": enhanced_results.get("total_results", 0),
            "context_chunks": enhanced_results.get("context_chunks_count", 0),
            "translated_query": translated_query,
            "has_context": bool(enhanced_results.get("context_text", "").strip()),
            "memory_enabled": True,
            "memory_type": "simple_in_memory",
            "conversation_length": len(conversation_history)
        }
        
        logger.info(f"Successfully processed question with {final_response['debug_info']['total_search_results']} search results")
        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return {
            "response": "Something went wrong while answering your question. Please try again later.",
            "buttons": False,
            "button_type": None,
            "button_data": None,
            "conversation_id": None,
            "debug_info": {
                "error": str(e),
                "total_search_results": 0,
                "context_chunks": 0,
                "has_context": False,
                "memory_enabled": False,
                "memory_type": "simple_in_memory"
            }
        }
