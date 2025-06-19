import base64
from contextlib import asynccontextmanager
import io
import os
import re
import traceback
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, Security, UploadFile
from fastapi.responses import JSONResponse, Response
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

from server.exceptions import FailedToConvertPDFException
from server.logger import logger
from server.schemas import OutputFormat, ParseResponse
from server.security import check_api_key

app_data = {}

STORAGE_DIR = "./tmp/uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(STORAGE_DIR, exist_ok=True)
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(
    lifespan=lifespan,
    title="Albert Marker",
    description="""Albert Marker is a FastAPI server for the Marker library designed for production environments.
    It is a simple server that can be used to convert PDF files to text, images, and metadata.
    Inspired by the [official Marker server script](https://github.com/VikParuchuri/marker/blob/b8a4c5d8769ed40f83d0ac9b452e85532ac7cd47/marker/scripts/server.py).
    """,
)


@app.get(path="/health", tags=["Misc"], dependencies=[Security(check_api_key)])  # fmt: off
def health() -> Response:
    return Response(status_code=200)


page_range = Form(default="", description="Page range to convert, specify comma separated page numbers or ranges. Example: '0,5-10,20'")  # fmt: off
force_ocr = Form(default=False, description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases).")  # fmt: off
paginate_output = Form(default=False, description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines).")  # fmt: off
output_format = Form(default="markdown", description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'.")  # fmt: off
use_llm = Form(default=False, description="Use LLM to improve conversion accuracy. Requires API key if using external services.")  # fmt: off
file = File(..., description="The PDF file to convert.")  # fmt: off


@app.post("/marker/upload", tags=["Marker"], response_model=ParseResponse, dependencies=[Security(check_api_key)])
async def convert_pdf_upload(
    page_range: Optional[str] = page_range,
    force_ocr: Optional[bool] = force_ocr,
    paginate_output: Optional[bool] = paginate_output,
    output_format: Optional[OutputFormat] = output_format,
    use_llm: Optional[bool] = use_llm,
    file: UploadFile = file,
) -> JSONResponse:
    """
    Parse a PDF file and return the text, images, and metadata.
    """
    # Validate page_range format if provided
    if page_range and not re.match(r"^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$", page_range):
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid page_range format. Use comma-separated numbers or ranges (e.g., '0,5-10,20') or leave blank for all pages."},
        )

    filepath = os.path.join(STORAGE_DIR, f"{uuid4()}_{file.filename}")
    with open(filepath, "wb+") as file_obj:
        file_contents = await file.read()
        file_obj.write(file_contents)

    options = {
        "filepath": filepath,
        "page_range": page_range,
        "force_ocr": force_ocr,
        "paginate_output": paginate_output,
        "output_format": output_format,
        "use_llm": use_llm,
    }
    try:
        config_parser = ConfigParser(cli_options=options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter = PdfConverter(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if options["use_llm"] else None,
        )
        rendered = converter(options["filepath"])
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata

    except Exception as e:
        logger.debug(traceback.format_exc())
        raise FailedToConvertPDFException(detail=str(e))

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(settings.OUTPUT_ENCODING)

    os.remove(filepath)

    response = ParseResponse(
        format=options["output_format"],
        output=text,
        images=encoded,
        metadata=metadata,
        success=True,
    )

    return JSONResponse(response.model_dump())
