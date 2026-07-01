from  dataclasses import dataclass, field
from  typing import Generic, TypeVar, Any


T = TypeVar("T")


@dataclass(slots=True)
class Error:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Result(Generic[T]):
    data: T | None = None
    error: Error | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_failure(self) -> bool:
        return self.error is not None

    @classmethod
    def ok(cls, data: T) -> "Result[T]":
        return cls(
            data=data,
        )

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "Result[T]":
        return cls(
            error=Error(
                code=code,
                message=message,
                details=details or {},
            )
        )