from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class AddUserRequest(BaseModel):
    username: str
    password: str


class ProcessUrlRequest(BaseModel):
    source_url: str
    transcription_model: str | None = None
    enable_transcription: bool = True
    enable_summary: bool = True
    enable_diarization: bool = True
    diarization_speakers: int = 0
