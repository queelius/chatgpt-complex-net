def messages_to_transcript(msgs, valid_roles=None):
    """
    Convert a list of message dicts to a conversation transcript string.
    Only include messages with roles in valid_roles (if specified).
    """

    lines = []
    for msg in msgs:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '').strip()
        if valid_roles is not None and msg.get('role') not in valid_roles:
            continue
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
