from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from app.auth.auth import get_current_active_user
from app.db.mysql import get_db
from app.utils.process_files import process_pdf, process_svg, process_text_file
from app.utils.common import create_chunks
from app.services.embeddings import get_embeddings
from app.db.qdrant import ingest_to_qdrant
from typing import Optional
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])

# Allowed file types
ALLOWED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'svg': 'image/svg+xml',
    'txt': 'text/plain',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

@router.post("/upload-and-process")
async def upload_and_process(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),  # Accept as form data
    current_user = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = None,
    db = Depends(get_db)
):
    """Upload and process files (PDF, SVG, etc.) and store in specified collection."""
    try:
        # If no collection_name provided, use user's default collection
        if not collection_name:
            collection_name = f"{current_user['id']}_default"
        
        logger.info(f"Starting file upload process for file: {file.filename} to collection: {collection_name}")
        
        # Validate file type
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
            )

        # Read file content
        try:
            file_content = await file.read()
            if not file_content:
                raise HTTPException(status_code=400, detail="Empty file received")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
        
        # Process file based on type
        text_content = ""
        try:
            if file_extension == 'pdf':
                text_content = process_pdf(file_content)
            elif file_extension == 'svg':
                text_content = process_svg(file_content)
            else:
                text_content = process_text_file(file_content)
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing file content: {str(e)}")

        if not text_content.strip():
            logger.warning("No text content extracted from file")
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file")

        # Create chunks from the text
        try:
            chunks = create_chunks(text_content, chunk_size=1000, overlap=200)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Validate chunks
            valid_chunks = [chunk for chunk in chunks if chunk.strip()]
            if len(valid_chunks) != len(chunks):
                logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks")
                chunks = valid_chunks
                
            if not chunks:
                raise HTTPException(status_code=400, detail="No valid text chunks could be created from the file")
                
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating text chunks: {str(e)}")

        # Generate embeddings
        try:
            embeddings = get_embeddings(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                raise Exception("Embedding generation failed or produced mismatched results")
            logger.info(f"Generated {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

        # Ingest to Qdrant
        try:
            ingest_to_qdrant(collection_name, chunks, embeddings)
            logger.info(f"Successfully ingested {len(chunks)} chunks to collection {collection_name}")
        except Exception as e:
            logger.error(f"Error ingesting to Qdrant: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error storing data: {str(e)}")

        return {
            "status": "success",
            "message": "File processed and stored successfully",
            "collection_name": collection_name,
            "chunks_created": len(chunks),
            "file_name": file.filename
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_and_process: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
