import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from services.object_detection_service import ObjectDetectionService

router = APIRouter()


@router.post("/", tags=["predict detect"], status_code=200)
def predict_and_detect(file: UploadFile):
    suffix = Path(file.filename).suffix

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)

        object_detection_service = ObjectDetectionService()
        bounding_boxes = object_detection_service.predict_and_detect(tmp.name)

        return JSONResponse(content=bounding_boxes)
