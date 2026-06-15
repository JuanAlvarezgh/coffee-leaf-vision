"""Custom FastAPI exception handlers."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


class ImageValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


async def image_validation_handler(request: Request, exc: ImageValidationError) -> JSONResponse:
    logger.warning(f"Image validation error: {exc.message}")
    return JSONResponse(status_code=400, content={"error": "Invalid image", "detail": exc.message})


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"error": "Validation error", "detail": str(exc)})


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )
