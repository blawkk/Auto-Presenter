from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from f5_tts.api import F5TTS
import shutil
import uuid
import os
import asyncio
from pathlib import Path

app = FastAPI()
f5tts = F5TTS()

current_task = None
task_lock = asyncio.Lock()

@app.post("/infer")
async def generate_audio(
    ref_file: UploadFile = File(...),
    ref_text: str = Form(...),
    gen_text: str = Form(...),
):
    global current_task
    
    async with task_lock:
        if current_task is not None and not current_task.done():
            current_task.cancel()
        
        current_task = asyncio.current_task()
    
    temp_filename = f"temp_{uuid.uuid4()}.wav"
    output_filename = f"generated_{uuid.uuid4()}.wav"
    
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(ref_file.file, buffer)
        
        await asyncio.to_thread(
            f5tts.infer,
            ref_file=temp_filename,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=output_filename,
            seed=None,
            nfe_step=16
        )
        
        return FileResponse(
            path=output_filename,
            filename=output_filename,
            media_type="audio/wav",
        )
    
    except asyncio.CancelledError:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)