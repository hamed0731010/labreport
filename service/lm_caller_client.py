import os
import json
import logging
import time
import requests
from typing import Optional, Dict, Any
from src.core.config import LLM_BASE_URL, LLM_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

class LmCallerClient:
    def __init__(self, base_url: str = LLM_BASE_URL):
        self.base_url = base_url.rstrip("/")
        # Updated endpoint to /run based on the curl command
        self.run_url = f"{self.base_url}/run"

    def extract_from_file(self, 
                          file_path: str, 
                          prompt_text: str, 
                          model_name: str, 
                          custom_schema: Dict[str, Any],
                          content_type: str = "raw_text",
                          debug: bool = False) -> Dict[str, Any]:
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.extract(
            content=content,
            content_type=content_type,
            prompt_text=prompt_text,
            model_name=model_name,
            custom_schema=custom_schema,
            filename=os.path.basename(file_path),
            debug=debug
        )
    
    def extract_from_text(self,
                          text: str,
                          prompt_text: str,
                          model_name: str,
                          custom_schema: Dict[str, Any],
                          filename: str = "text_input.txt",
                          debug: bool = False) -> Dict[str, Any]:
        
        return self.extract(
            content=text,
            content_type="raw_text",
            prompt_text=prompt_text,
            model_name=model_name,
            custom_schema=custom_schema,
            filename=filename,
            debug=debug
        )
    
    def extract_from_image_b64(self,
                               image_b64: str,
                               prompt_text: str,
                               model_name: str,
                               custom_schema: Dict[str, Any],
                               filename: str = "image.png",
                               debug: bool = False) -> Dict[str, Any]:
        
        return self.extract(
            content=image_b64,
            content_type="image_b64",
            prompt_text=prompt_text,
            model_name=model_name,
            custom_schema=custom_schema,
            filename=filename,
            debug=debug
        )
    
    def extract(self, 
                content: str,
                content_type: str,
                prompt_text: str, 
                model_name: str, 
                custom_schema: Dict[str, Any],
                filename: str = "unknown",
                debug: bool = False) -> Dict[str, Any]:
        
        # Validate content_type
        if content_type not in ['raw_text', 'image_b64']:
            raise ValueError("content_type must be either 'raw_text' or 'image_b64'")
        
        # 1. Prepare raw dictionary
        raw_data = {
            "content": content,
            "prompt_text": prompt_text,
            "model_name": model_name,
            "custom_schema": json.dumps(custom_schema) if isinstance(custom_schema, dict) else custom_schema,
            "filename": filename,
        }
        
        # Remove None values
        raw_data = {k: v for k, v in raw_data.items() if v is not None}

        # 2. Format as Multipart Form Data to match `curl --form`
        multipart_data = {
            key: (None, str(value).encode('utf-8'))
            for key, value in raw_data.items()
        }

        if debug:
            print("\n" + "="*60)
            print("DEBUG: Request Details")
            print(f"URL: {self.run_url}")
            print(f"Method: POST (multipart/form-data)")
            print("="*60 + "\n")

        try:
            # 3. Use 'files=' instead of 'data=' to force multipart boundary creation
            started_at = time.monotonic()
            logger.info(
                "Calling LLM runner for %s (read timeout %.0f seconds)",
                filename, LLM_REQUEST_TIMEOUT,
            )
            response = requests.post(
                self.run_url,
                files=multipart_data,
                headers={"accept": "application/json"},
                timeout=(min(10.0, LLM_REQUEST_TIMEOUT), LLM_REQUEST_TIMEOUT)
            )
            logger.info(
                "LLM runner response for %s: HTTP %s in %.1f seconds",
                filename,
                response.status_code,
                time.monotonic() - started_at,
            )
            
            if not response.ok:
                print("\n" + "="*60)
                print(f"ERROR: Server returned {response.status_code}")
                print(f"Response Body (Text): {response.text[:2000]}")
                print("="*60 + "\n")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP Error: {e} - Response Body: {response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {e}")
