from llamda_fn.llms.ll_message import OaiRole

__all__: list[str] = ["get_role_emoji", "role_emojis"]

role_emojis: dict[OaiRole, str] = {
    "user": "🐱",
    "assistant": "🐙",
    "tool": "🧑🏼‍🔧",
    "system": "📐",
}


def get_role_emoji(role: OaiRole) -> str:
    """Get the emoji for the role"""
    if role not in role_emojis:
        raise ValueError(f"Role {role} not supported")
    return role_emojis[role]
