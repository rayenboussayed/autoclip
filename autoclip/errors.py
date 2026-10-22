"""Exception hierarchy: everything the pipeline raises on purpose derives from :class:`AutoclipError`."""


class AutoclipError(Exception):
    """A problem with one video or one clip that should be reported, not crash the whole batch."""


class ConfigError(AutoclipError):
    """The configuration file (or an input path it names) is invalid."""
