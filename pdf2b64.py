# pdf2b64.py - Updated with better error handling and Poppler path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import os
import sys
import traceback

# Try to add Poppler to PATH if on Windows
if sys.platform == "win32":
    poppler_paths = [
        r"C:\poppler\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\poppler-24.08.0\Library\bin",  # Version-specific path
    ]
    for path in poppler_paths:
        if os.path.exists(path):
            os.environ["PATH"] = path + ";" + os.environ["PATH"]
            print(f"Added Poppler to PATH: {path}")
            break

try:
    from pdf2image import convert_from_bytes

    print("✅ pdf2image imported successfully")
except ImportError as e:
    print(f"❌ Failed to import pdf2image: {e}")
    print("Run: pip install pdf2image")

import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "PDF to Base64 API is running!"}


@app.post("/convert/")
async def convert_pdf_to_images(file: UploadFile = File(...)):
    try:
        # Read the uploaded PDF file
        pdf_content = await file.read()
        print(f"Received PDF file: {file.filename}, Size: {len(pdf_content)} bytes")

        # Try to convert PDF to images
        try:
            images = convert_from_bytes(pdf_content)
            print(f"Successfully converted PDF to {len(images)} images")
        except Exception as pdf_error:
            print(f"PDF conversion error: {pdf_error}")
            error_msg = str(pdf_error)

            if (
                "Unable to get page count" in error_msg
                or "PDFInfoNotInstalledError" in str(type(pdf_error))
            ):
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Poppler not found. Please install Poppler for Windows.",
                        "error": error_msg,
                        "instructions": [
                            "1. Download from: https://github.com/oschwartz10612/poppler-windows/releases",
                            "2. Extract to C:\\poppler",
                            "3. Add C:\\poppler\\Library\\bin to PATH",
                            "4. Restart this server",
                        ],
                    },
                )
            raise

        # Convert images to Base64
        base64_images = []
        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
            print(f"Converted image {i+1} to base64 (size: {len(img_str)} chars)")

        return JSONResponse(content={"images": base64_images})

    except Exception as e:
        print(f"Error in convert_pdf_to_images: {e}")
        print(f"Error type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": str(type(e).__name__),
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Starting FastAPI server...")
    print("Access the API at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("=" * 60)

    # Test if Poppler is available
    try:
        from pdf2image import convert_from_bytes

        test_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test PDF) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000262 00000 n\n0000000341 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n436\n%%EOF"
        images = convert_from_bytes(test_pdf)
        print("✅ Poppler is properly installed and working!")
    except Exception as e:
        print("❌ Poppler is not properly installed!")
        print(f"Error: {e}")
        print("\nPlease install Poppler:")
        print("1. Download: https://github.com/oschwartz10612/poppler-windows/releases")
        print("2. Extract to C:\\poppler")
        print("3. Add C:\\poppler\\Library\\bin to PATH")

    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)