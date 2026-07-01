# bootstrap/lifecycle.py

from  abc import ABC
from  abc import abstractmethod


class Lifecycle(ABC):

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass