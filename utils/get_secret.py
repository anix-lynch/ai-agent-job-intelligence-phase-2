# Backward compatibility: use shared.get_secret
from shared.get_secret import get_secret, load_secrets_to_env
__all__ = ["get_secret", "load_secrets_to_env"]
