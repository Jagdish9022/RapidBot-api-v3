from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.db.models import ScrapeRequest
from app.auth.auth import get_current_active_user
from app.db.mysql import get_db
from app.utils.task_management import (
    get_progress_safely, update_progress_safely, 
    scraping_progress, progress_lock
)
from app.utils.scraping_utils import process_scraping_sync
from app.db.qdrant import request_cancellation, clear_cancellation_request
import logging
from datetime import datetime
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["scraping"])

# Thread pool for concurrent scraping tasks
scraping_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="scraper")

@router.post("/scrape-and-ingest")
async def scrape_and_ingest(
    req: ScrapeRequest,
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Scrape a website and ingest the content into specified collection."""
    try:
        # Use collection_name from request (now required from frontend)
        collection_name = req.collection_name
        
        logger.info(f"Starting scrape and ingest for URL: {req.url} to collection: {collection_name}")
        
        # Generate a unique ID for this scraping task
        task_id = hashlib.md5(f"{req.url}_{collection_name}_{datetime.now().timestamp()}".encode()).hexdigest()
        
        # Initialize progress tracking with thread safety
        with progress_lock:
            scraping_progress[task_id] = {
                "status": "queued",
                "start_time": datetime.now(),
                "last_update": datetime.now(),
                "pages_scraped": 0,
                "chunks_created": 0,
                "error": None,
                "is_completed": False,
                "collection_name": collection_name,
                "url": req.url,
                "user_id": current_user['id']  # Track which user started this task
            }
        
        # Submit task to thread pool executor
        future = scraping_executor.submit(process_scraping_sync, req.url, task_id, collection_name)
        
        # Don't wait for the result, just return the task_id immediately
        logger.info(f"Scraping task {task_id} submitted to thread pool")
        
        return {
            "task_id": task_id, 
            "status": "queued",
            "collection_name": collection_name,
            "message": "Scraping task has been queued and will start shortly"
        }
        
    except Exception as e:
        logger.error(f"Error starting scrape process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scraping-progress/{task_id}")
async def get_scraping_progress(
    task_id: str,
    current_user = Depends(get_current_active_user)
):
    """Get the current progress of a scraping task."""
    progress_data = get_progress_safely(task_id)
    
    if not progress_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if the current user owns this task (optional security check)
    if progress_data.get("user_id") != current_user['id']:
        raise HTTPException(status_code=403, detail="Access denied to this task")
    
    # Check if we should return cached response
    current_time = datetime.now()
    last_update = progress_data["last_update"]
    time_diff = (current_time - last_update).total_seconds()
    
    # If the task is completed or there's an error, return immediately
    if progress_data.get("is_completed", False) or progress_data.get("error"):
        return progress_data
    
    # If less than 2 seconds have passed since last update, return cached response
    if time_diff < 2:
        return progress_data
    
    # Update last access time (but don't change the actual last_update from the worker)
    return progress_data

@router.get("/scraping-tasks")
async def get_user_scraping_tasks(
    current_user = Depends(get_current_active_user),
    include_completed: bool = False
):
    """Get all scraping tasks for the current user."""
    user_tasks = {}
    
    with progress_lock:
        for task_id, task_data in scraping_progress.items():
            if task_data.get("user_id") == current_user['id']:
                if include_completed or not task_data.get("is_completed", False):
                    user_tasks[task_id] = task_data.copy()
    
    return {
        "tasks": user_tasks,
        "total_count": len(user_tasks)
    }

@router.post("/stop-scraping/{task_id}")
async def stop_scraping(
    task_id: str,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Stop a running scraping task and deactivate associated chatbot."""
    try:
        logger.info(f"Stop scraping request for task {task_id} by user {current_user['id']}")
        
        # Check if task exists in progress tracking
        with progress_lock:
            if task_id not in scraping_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Scraping task {task_id} not found"
                )
            
            # Check if user has permission to stop this task
            task_info = scraping_progress[task_id]
            if task_info.get("user_id") != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to stop task {task_id} owned by user {task_info.get('user_id')}")
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to stop this task"
                )
            
            # Check if task is already completed or cancelled
            current_status = task_info.get("status", "unknown")
            if current_status in ["completed", "cancelled", "error"]:
                logger.info(f"Task {task_id} is already in final state: {current_status}")
                return {
                    "message": f"Task {task_id} is already {current_status}",
                    "status": current_status,
                    "task_id": task_id
                }
            
            # Get collection name from task info for chatbot lookup
            collection_name = task_info.get("collection_name")
        
        # Request cancellation in the Qdrant module
        request_cancellation(task_id)
        logger.info(f"Cancellation requested for task {task_id}")
        
        # Complete the cancellation process directly
        cancellation_time = datetime.now()
        
        # Update progress to cancelled state immediately
        update_progress_safely(task_id, "cancelled", 
                         error="Task cancelled by user",
                         cancellation_time=cancellation_time,
                         is_completed=True)
        
        # Update chatbot is_active to false if collection_name exists
        chatbot_updated = False
        if collection_name:
            try:
                cursor = db.cursor()
                cursor.execute("""
                    UPDATE chatbots 
                    SET is_active = FALSE, updated_at = %s
                    WHERE collection_name = %s AND user_id = %s AND is_active = TRUE
                """, (cancellation_time, collection_name, current_user['id']))
                
                db.commit()
                
                if cursor.rowcount > 0:
                    chatbot_updated = True
                    logger.info(f"Chatbot with collection '{collection_name}' deactivated for user {current_user['id']}")
                else:
                    logger.info(f"No active chatbot found with collection '{collection_name}' for user {current_user['id']}")
                
                cursor.close()
                
            except Exception as db_error:
                logger.error(f"Error updating chatbot status for collection '{collection_name}': {db_error}")
                # Don't raise the exception here as the main scraping cancellation was successful
        
        logger.info(f"Scraping task {task_id} has been successfully cancelled and cleaned up")
        
        response = {
            "message": f"Scraping task {task_id} has been successfully cancelled",
            "status": "cancelled",
            "task_id": task_id,
            "timestamp": cancellation_time.isoformat()
        }
        
        # Add chatbot update status to response
        if collection_name:
            response["chatbot_deactivated"] = chatbot_updated
            response["collection_name"] = collection_name
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error stopping scraping task {task_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to stop scraping task: {str(e)}"
        )

@router.post("/stop-and-store-scraping/{task_id}")
async def stop_and_store_scraping(
    task_id: str,
    current_user = Depends(get_current_active_user),
    db = Depends(get_db)
):
    """Stop a running scraping task and store only the pages crawled so far."""
    try:
        logger.info(f"Stop and store scraping request for task {task_id} by user {current_user['id']}")
        
        # Check if task exists in progress tracking
        with progress_lock:
            if task_id not in scraping_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Scraping task {task_id} not found"
                )
            
            # Check if user has permission to stop this task
            task_info = scraping_progress[task_id]
            if task_info.get("user_id") != current_user['id']:
                logger.warning(f"User {current_user['id']} attempted to stop task {task_id} owned by user {task_info.get('user_id')}")
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to stop this task"
                )
            
            # Check if task is already completed or cancelled
            current_status = task_info.get("status", "unknown")
            if current_status in ["completed", "cancelled", "error"]:
                logger.info(f"Task {task_id} is already in final state: {current_status}")
                return {
                    "message": f"Task {task_id} is already {current_status}",
                    "status": current_status,
                    "task_id": task_id,
                    "pages_stored": task_info.get("pages_scraped", 0),
                    "chunks_stored": task_info.get("chunks_created", 0)
                }
            
            # Get collection name and current progress
            collection_name = task_info.get("collection_name")
            pages_scraped = task_info.get("pages_scraped", 0)
            chunks_created = task_info.get("chunks_created", 0)
        
        # Request cancellation but mark for partial storage
        request_cancellation(task_id)
        logger.info(f"Cancellation with partial storage requested for task {task_id}")
        
        # Update status to indicate we're stopping and storing partial data
        update_progress_safely(task_id, "stopping_and_storing", 
                             message="Stopping scraping and storing crawled pages...")
        
        # Wait a brief moment for the scraping thread to process the cancellation
        # and complete any partial ingestion
        import asyncio
        await asyncio.sleep(2)
        
        # Check final status after cancellation processing
        with progress_lock:
            final_task_info = scraping_progress.get(task_id, {})
            final_pages = final_task_info.get("pages_scraped", pages_scraped)
            final_chunks = final_task_info.get("chunks_created", chunks_created)
            ingestion_result = final_task_info.get("result", {})
        
        # Complete the cancellation process
        cancellation_time = datetime.now()
        
        # Update progress to partial completion state
        update_progress_safely(task_id, "partial_completed", 
                             message=f"Scraping stopped. Stored {final_pages} pages with {final_chunks} chunks",
                             cancellation_time=cancellation_time,
                             is_completed=True)
        
        # Update chatbot to active if we have stored data and collection exists
        chatbot_updated = False
        if collection_name and final_chunks > 0:
            try:
                cursor = db.cursor()
                
                # Check if chatbot already exists for this collection
                cursor.execute("""
                    SELECT id FROM chatbots 
                    WHERE collection_name = %s AND user_id = %s
                """, (collection_name, current_user['id']))
                
                existing_chatbot = cursor.fetchone()
                
                if existing_chatbot:
                    # Update existing chatbot to active
                    cursor.execute("""
                        UPDATE chatbots 
                        SET is_active = TRUE, updated_at = %s
                        WHERE collection_name = %s AND user_id = %s
                    """, (cancellation_time, collection_name, current_user['id']))
                    chatbot_updated = True
                    logger.info(f"Chatbot with collection '{collection_name}' activated for partial data")
                else:
                    # Create new chatbot entry for the partial data
                    cursor.execute("""
                        INSERT INTO chatbots (user_id, name, collection_name, is_active, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        current_user['id'], 
                        f"Partial Scrape - {collection_name}", 
                        collection_name, 
                        True, 
                        cancellation_time, 
                        cancellation_time
                    ))
                    chatbot_updated = True
                    logger.info(f"New chatbot created for partial collection '{collection_name}'")
                
                db.commit()
                cursor.close()
                
            except Exception as db_error:
                logger.error(f"Error updating chatbot status for collection '{collection_name}': {db_error}")
                # Don't raise the exception here as the main operation was successful
        
        logger.info(f"Scraping task {task_id} stopped and {final_pages} pages stored successfully")
        
        response = {
            "message": f"Scraping stopped and partial data stored successfully",
            "status": "partial_completed",
            "task_id": task_id,
            "pages_stored": final_pages,
            "chunks_stored": final_chunks,
            "collection_name": collection_name,
            "timestamp": cancellation_time.isoformat()
        }
        
        # Add ingestion details if available
        if ingestion_result:
            response["ingestion_details"] = {
                "total_points": ingestion_result.get("total_points", 0),
                "ingested_points": ingestion_result.get("ingested_points", 0),
                "ingestion_status": ingestion_result.get("status", "unknown")
            }
        
        # Add chatbot status to response
        if collection_name:
            response["chatbot_available"] = chatbot_updated and final_chunks > 0
            if not chatbot_updated and final_chunks > 0:
                response["chatbot_note"] = "Data stored but chatbot activation failed"
            elif final_chunks == 0:
                response["chatbot_note"] = "No data stored, chatbot not available"
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error stopping and storing scraping task {task_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to stop and store scraping task: {str(e)}"
        )

@router.delete("/scraping-tasks/{task_id}")
async def delete_scraping_task(
    task_id: str,
    current_user = Depends(get_current_active_user)
):
    """Delete a scraping task from progress tracking (for completed/cancelled tasks)."""
    try:
        logger.info(f"Delete task {task_id} request by user {current_user['id']}")
        
        with progress_lock:
            if task_id not in scraping_progress:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Task {task_id} not found"
                )
            
            task_info = scraping_progress[task_id]
            
            # Check user permission
            if task_info.get("user_id") != current_user['id']:
                raise HTTPException(
                    status_code=403, 
                    detail="You don't have permission to delete this task"
                )
            
            # Only allow deletion of completed/cancelled/error tasks
            status = task_info.get("status", "unknown")
            if status not in ["completed", "cancelled", "error", "partial_completed"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete active task. Current status: {status}. Stop the task first."
                )
            
            # Remove from progress tracking
            del scraping_progress[task_id]
            
        # Also clear any remaining cancellation requests
        clear_cancellation_request(task_id)
        
        logger.info(f"Task {task_id} deleted successfully")
        
        return {
            "message": f"Task {task_id} deleted successfully",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to delete task: {str(e)}"
        )

@router.get("/scraping-tasks")
async def get_all_scraping_tasks(
    current_user = Depends(get_current_active_user)
):
    """Get all scraping tasks with their current status."""
    try:
        # Filter tasks and add additional info
        tasks_info = []
        current_time = datetime.now()
        
        for task_id, task_data in scraping_progress.items():
            task_info = {
                "task_id": task_id,
                "status": task_data["status"],
                "collection_name": task_data.get("collection_name", "unknown"),
                "url": task_data.get("url", "unknown"),
                "start_time": task_data["start_time"].isoformat(),
                "last_update": task_data["last_update"].isoformat(),
                "pages_scraped": task_data.get("pages_scraped", 0),
                "chunks_created": task_data.get("chunks_created", 0),
                "is_completed": task_data["is_completed"],
                "error": task_data.get("error"),
                "duration_seconds": (current_time - task_data["start_time"]).total_seconds(),
                "can_be_cancelled": task_data["status"] in ["crawling", "processing", "generating_embeddings", "storing"]
            }
            tasks_info.append(task_info)
        
        # Sort by start time (newest first)
        tasks_info.sort(key=lambda x: x["start_time"], reverse=True)
        
        return {
            "tasks": tasks_info,
            "total_count": len(tasks_info),
            "active_count": len([t for t in tasks_info if not t["is_completed"]]),
            "completed_count": len([t for t in tasks_info if t["is_completed"] and not t["error"]]),
            "error_count": len([t for t in tasks_info if t["error"]])
        }
        
    except Exception as e:
        logger.error(f"Error fetching scraping tasks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error fetching scraping tasks"
        )
