import os
import re
import shlex
from enum import Enum
from pathlib import Path
from typing import List
from pydantic import BaseModel


class ResourceType(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    PORT = "port"
    PROCESS = "process"
    PACKAGE_MANAGER = "package_manager"
    GIT_REPO = "git_repo"
    URL = "url"


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


class ResourceRequirement(BaseModel):
    resource_type: ResourceType
    identifier: str
    access_mode: AccessMode


READ_COMMANDS = {
    "cat",
    "less",
    "head",
    "tail",
    "grep",
    "find",
    "ls",
}


WRITE_COMMANDS = {
    "touch",
    "echo",
    "sed",
    "tee",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "truncate",
}


PACKAGE_MANAGERS = {
    "npm",
    "yarn",
    "pnpm",
    "pip",
    "apt",
    "apt-get",
    "cargo",
    "poetry",
}


GIT_COMMANDS = {
    "git",
}


FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".toml",
    ".ini",
    ".env",
    ".csv",
    ".xml",
    ".html",
    ".css",
}


PORT_PATTERNS = [
    r"--port\s+(\d+)",
    r":(\d+)",
    r"port\s+(\d+)",
]


URL_PATTERN = r"https?://[^\s]+"


PATH_PATTERN = re.compile(r"(?:\./|\.\./|/)?(?:[\w\-.]+/)*[\w\-.]+(?:\.[\w]+)?")


REDIRECTION_OPERATORS = {
    ">",
    ">>",
}


DESTRUCTIVE_COMMANDS = {
    "rm",
    "mv",
}


class ResourceDetector:

    @staticmethod
    def detect(command: str) -> List[ResourceRequirement]:
        resources: List[ResourceRequirement] = []

        try:
            tokens = shlex.split(command)
        except Exception:
            return resources

        if not tokens:
            return resources

        base_command = tokens[0]

        # --------------------------------------------------
        # PACKAGE MANAGERS
        # --------------------------------------------------

        if base_command in PACKAGE_MANAGERS:
            resources.append(
                ResourceRequirement(
                    resource_type=ResourceType.PACKAGE_MANAGER,
                    identifier=base_command,
                    access_mode=AccessMode.EXCLUSIVE,
                )
            )

            # node_modules is almost always shared mutable state
            if base_command in {"npm", "yarn", "pnpm"}:
                resources.append(
                    ResourceRequirement(
                        resource_type=ResourceType.DIRECTORY,
                        identifier="./node_modules",
                        access_mode=AccessMode.WRITE,
                    )
                )

        # --------------------------------------------------
        # GIT
        # --------------------------------------------------

        if base_command in GIT_COMMANDS:
            resources.append(
                ResourceRequirement(
                    resource_type=ResourceType.GIT_REPO,
                    identifier=os.getcwd(),
                    access_mode=AccessMode.EXCLUSIVE,
                )
            )

        # --------------------------------------------------
        # URLS
        # --------------------------------------------------

        urls = re.findall(URL_PATTERN, command)

        for url in urls:
            resources.append(
                ResourceRequirement(
                    resource_type=ResourceType.URL,
                    identifier=url,
                    access_mode=AccessMode.READ,
                )
            )

        # --------------------------------------------------
        # PORTS
        # --------------------------------------------------

        for pattern in PORT_PATTERNS:
            matches = re.findall(pattern, command)

            for port in matches:
                resources.append(
                    ResourceRequirement(
                        resource_type=ResourceType.PORT,
                        identifier=port,
                        access_mode=AccessMode.EXCLUSIVE,
                    )
                )

        # --------------------------------------------------
        # FILES / DIRECTORIES
        # --------------------------------------------------

        for i, token in enumerate(tokens):

            # Skip shell operators
            if token in {"|", "&&", "||"}:
                continue

            # --------------------------------------------------
            # Detect redirection writes
            # --------------------------------------------------

            if token in REDIRECTION_OPERATORS:
                if i + 1 < len(tokens):
                    path = tokens[i + 1]

                    resources.append(
                        ResourceRequirement(
                            resource_type=ResourceType.FILE,
                            identifier=path,
                            access_mode=AccessMode.WRITE,
                        )
                    )

                continue

            # --------------------------------------------------
            # Path heuristic
            # --------------------------------------------------

            if ResourceDetector._looks_like_path(token):

                access_mode = ResourceDetector._infer_access_mode(
                    base_command=base_command,
                    token_index=i,
                    tokens=tokens,
                )

                resource_type = ResourceDetector._infer_resource_type(token)

                resources.append(
                    ResourceRequirement(
                        resource_type=resource_type,
                        identifier=token,
                        access_mode=access_mode,
                    )
                )

        # --------------------------------------------------
        # Deduplicate
        # --------------------------------------------------

        return ResourceDetector._deduplicate(resources)

    @staticmethod
    def _looks_like_path(token: str) -> bool:

        if token.startswith("-"):
            return False

        if token.startswith("http://") or token.startswith("https://"):
            return False

        if any(token.endswith(ext) for ext in FILE_EXTENSIONS):
            return True

        if "/" in token:
            return True

        if token.startswith("."):
            return True

        return False

    @staticmethod
    def _infer_resource_type(path: str) -> ResourceType:

        p = Path(path)

        if p.suffix:
            return ResourceType.FILE

        return ResourceType.DIRECTORY

    @staticmethod
    def _infer_access_mode(
        base_command: str,
        token_index: int,
        tokens: List[str],
    ) -> AccessMode:

        if base_command in READ_COMMANDS:
            return AccessMode.READ

        if base_command in DESTRUCTIVE_COMMANDS:
            return AccessMode.EXCLUSIVE

        if base_command in WRITE_COMMANDS:
            return AccessMode.WRITE

        # fallback heuristic
        previous_token = tokens[token_index - 1] if token_index > 0 else ""

        if previous_token in REDIRECTION_OPERATORS:
            return AccessMode.WRITE

        return AccessMode.READ

    @staticmethod
    def _deduplicate(
        resources: List[ResourceRequirement],
    ) -> List[ResourceRequirement]:

        unique = {}

        priority = {
            AccessMode.READ: 1,
            AccessMode.WRITE: 2,
            AccessMode.EXCLUSIVE: 3,
        }

        for resource in resources:
            key = (
                resource.resource_type,
                resource.identifier,
            )

            existing = unique.get(key)

            if existing is None:
                unique[key] = resource
                continue

            if priority[resource.access_mode] > priority[existing.access_mode]:
                unique[key] = resource

        return list(unique.values())


FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".toml",
    ".ini",
    ".env",
    ".csv",
    ".xml",
    ".html",
    ".css",
}


SHELL_OPERATORS = {
    "|",
    "&&",
    "||",
    ">",
    ">>",
    "<",
    "<<",
}


def looks_like_path(token: str) -> bool:
    """
    Heuristic to determine whether a token
    looks like a filesystem path.
    """

    # flags/options
    if token.startswith("-"):
        return False

    # urls
    if token.startswith("http://"):
        return False

    if token.startswith("https://"):
        return False

    # relative/absolute paths
    if token.startswith("/"):
        return True

    if token.startswith("./"):
        return True

    if token.startswith("../"):
        return True

    if "/" in token:
        return True

    # file extensions
    if any(token.endswith(ext) for ext in FILE_EXTENSIONS):
        return True

    # hidden files
    if token.startswith("."):
        return True

    return False


def normalize_path(path: str) -> str:
    """
    Normalize filesystem path.
    """

    try:
        return str(Path(path).resolve())
    except Exception:
        return path


def extract_paths(command: str) -> List[str]:
    """
    Extract filesystem paths from a shell command.
    """

    paths = []

    try:
        tokens = shlex.split(command)
    except Exception:
        return []

    for token in tokens:

        # skip shell operators
        if token in SHELL_OPERATORS:
            continue

        if looks_like_path(token):

            normalized = normalize_path(token)

            paths.append(normalized)

    # deduplicate while preserving order
    unique_paths = list(dict.fromkeys(paths))

    return unique_paths


class ResourceDetails(BaseModel):
    type: str
    identifier: str


def detect_resources(command: str) -> List[ResourceDetails]:
    paths = extract_paths(command)
    resources = []

    for path in paths:
        resources.append(ResourceDetails(type="file", identifier=path))

    return resources
