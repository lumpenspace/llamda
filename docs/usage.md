
# Usage

## Basic Function with Multiple Arguments

```python
@llamda.llamdafy()
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width
```

## Function with Optional and Default Arguments

```python
from typing import Optional

@llamda.llamdafy()
def create_user(name: str, age: int, email: Optional[str] = None) -> dict:
    """Create a user with optional email."""
    return {"name": name, "age": age, "email": email}
```

## Function with Complex Types

```python
from typing import List, Dict

@llamda.llamdafy()
def process_data(items: List[str], settings: Dict[str, int]) -> List[Dict[str, Any]]:
    """Process a list of items based on settings."""
    return [{"item": item, "value": settings.get(item, 0)} for item in items]
```

## Using Pydantic Models

```python
from pydantic import BaseModel, Field

class UserModel(BaseModel):
    name: str
    age: int = Field(gt=0)
    email: Optional[str] = None

@llamda.llamdafy()
def create_user_from_model(user: UserModel) -> dict:
    """Create a user from a Pydantic model."""
    return user.model_dump()
```
