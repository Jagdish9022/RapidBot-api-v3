import time
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests

from app.utils.common import scrape_url, clean_text, create_chunks, should_skip_url
from app.db.qdrant import (
    ingest_to_qdrant_incremental, 
    is_cancellation_requested,
    CancellationException
)
from app.utils.task_management import update_progress_safely

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_scraping_sync(url: str, task_id: str, collection_name: str):
    """Process scraping with incremental storage - store data as it's crawled."""
    try:
        update_progress_safely(task_id, "starting", message="Initializing scraping process...")
        
        # Initialize counters
        total_pages_crawled = 0
        total_chunks_stored = 0
        
        # Initialize embeddings model
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Start crawling with incremental processing
        visited = set()
        urls_to_visit = [url]
        start_domain = urlparse(url).netloc
        
        while urls_to_visit and not is_cancellation_requested(task_id):
            current_url = urls_to_visit.pop(0)
            
            if current_url in visited:
                continue
                
            try:
                # Update progress
                update_progress_safely(task_id, "crawling", 
                                     message=f"Crawling page {total_pages_crawled + 1}: {current_url}",
                                     pages_scraped=total_pages_crawled)
                
                # Scrape current page
                html_content = scrape_url(current_url)
                if not html_content:
                    continue
                
                # Check if it's HTML content
                try:
                    response_check = requests.head(current_url, timeout=5)
                    content_type = response_check.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        continue
                except:
                    pass
                
                visited.add(current_url)
                total_pages_crawled += 1
                
                # IMMEDIATELY process this page's content
                clean_text_content = clean_text(html_content)
                if clean_text_content:
                    # Create chunks from this page
                    page_chunks = create_chunks(clean_text_content, chunk_size=1000, overlap=200)
                    
                    if page_chunks:
                        # Check for cancellation before processing chunks
                        if is_cancellation_requested(task_id):
                            logger.info(f"Task {task_id} cancelled during chunk processing for page {current_url}")
                            break
                        
                        # Generate embeddings for these chunks
                        try:
                            embeddings = embeddings_model.encode(page_chunks).tolist()
                            
                            # Store chunks immediately
                            result = ingest_to_qdrant_incremental(
                                collection_name=collection_name,
                                texts=page_chunks,
                                embeddings=embeddings,
                                task_id=task_id,
                                progress_callback=lambda stored, total, msg: update_progress_safely(
                                    task_id, "processing", 
                                    message=f"Stored {stored}/{total} chunks from page {total_pages_crawled}",
                                    pages_scraped=total_pages_crawled,
                                    chunks_created=total_chunks_stored + stored
                                )
                            )
                            
                            # Update chunk count
                            chunks_from_this_page = result.get("ingested_points", 0)
                            total_chunks_stored += chunks_from_this_page
                            
                            logger.info(f"Page {total_pages_crawled}: Stored {chunks_from_this_page} chunks. Total: {total_chunks_stored}")
                            
                        except Exception as e:
                            logger.error(f"Error processing chunks for {current_url}: {e}")
                            # Continue with next page even if this one fails
                
                # Extract links for further crawling (only if not cancelled)
                if not is_cancellation_requested(task_id):
                    soup = BeautifulSoup(html_content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        full_url = urljoin(current_url, link["href"])
                        
                        # Only crawl links from the same domain
                        if (urlparse(full_url).netloc == start_domain
                            and full_url not in visited
                            and full_url not in urls_to_visit
                            and not should_skip_url(full_url)):
                            urls_to_visit.append(full_url)
                
                # Update progress after each page
                update_progress_safely(task_id, "crawling", 
                                     message=f"Crawled {total_pages_crawled} pages, stored {total_chunks_stored} chunks",
                                     pages_scraped=total_pages_crawled,
                                     chunks_created=total_chunks_stored)
                
                # Small delay to make cancellation more responsive
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {current_url}: {e}")
                continue
        
        # Final status update
        if is_cancellation_requested(task_id):
            final_status = "cancelled"
            final_message = f"Scraping cancelled. Processed {total_pages_crawled} pages, stored {total_chunks_stored} chunks"
        else:
            final_status = "completed"
            final_message = f"Scraping completed. Processed {total_pages_crawled} pages, stored {total_chunks_stored} chunks"
        
        update_progress_safely(task_id, final_status,
                             message=final_message,
                             pages_scraped=total_pages_crawled,
                             chunks_created=total_chunks_stored,
                             is_completed=True,
                             result={
                                 "total_pages": total_pages_crawled,
                                 "total_chunks": total_chunks_stored,
                                 "collection_name": collection_name,
                                 "status": final_status
                             })
        
        logger.info(f"Task {task_id} {final_status}: {total_pages_crawled} pages, {total_chunks_stored} chunks")
        
    except CancellationException:
        logger.info(f"Task {task_id} was cancelled during processing")
        update_progress_safely(task_id, "cancelled",
                             message="Task cancelled by user",
                             pages_scraped=total_pages_crawled if 'total_pages_crawled' in locals() else 0,
                             chunks_created=total_chunks_stored if 'total_chunks_stored' in locals() else 0,
                             is_completed=True)
    except Exception as e:
        logger.error(f"Error in scraping task {task_id}: {e}")
        update_progress_safely(task_id, "error",
                             error=str(e),
                             pages_scraped=total_pages_crawled if 'total_pages_crawled' in locals() else 0,
                             chunks_created=total_chunks_stored if 'total_chunks_stored' in locals() else 0,
                             is_completed=True)
