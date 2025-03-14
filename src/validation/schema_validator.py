from typing import TypeVar, Generic, Dict, Any, Optional, List

T = TypeVar('T')

class SchemaValidator(Generic[T]):
    def __init__(self, schema: Dict[str, Any]):
        self.schema: Dict[str, Any] = schema
        self.errors: List[str] = []

    def validate(self, data: T) -> bool:
        self.errors = []
        return self._validate_object(data, self.schema)

    def _validate_object(self, obj: Any, schema: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            self.errors.append(f"Expected object, got {type(obj)}")
            return False
        
        result = True
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        
        # Check required fields
        for field in required_fields:
            if field not in obj:
                self.errors.append(f"Missing required field: {field}")
                result = False
        
        # Validate each field
        for field_name, field_value in obj.items():
            if field_name in properties:
                field_schema = properties[field_name]
                if not self.validate_field(field_value, field_schema):
                    result = False
            else:
                # Handle additional properties if specified
                additional_props = schema.get('additionalProperties', True)
                if additional_props is False:
                    self.errors.append(f"Unknown field: {field_name}")
                    result = False
                elif isinstance(additional_props, dict):
                    if not self.validate_field(field_value, additional_props):
                        result = False
        
        return result
    
    def validate_field(self, value: Any, field_schema: Dict[str, Any]) -> bool:
        field_type = field_schema.get('type')
        return self._check_type(value, field_type)
    
    def _check_type(self, value: Any, expected_type: Optional[str]) -> bool:
        if expected_type is None:
            return True
        
        if expected_type == 'string':
            if not isinstance(value, str):
                self.errors.append(f"Expected string, got {type(value)}")
                return False
        elif expected_type == 'number':
            if not isinstance(value, (int, float)):
                self.errors.append(f"Expected number, got {type(value)}")
                return False
        elif expected_type == 'integer':
            if not isinstance(value, int):
                self.errors.append(f"Expected integer, got {type(value)}")
                return False
        elif expected_type == 'boolean':
            if not isinstance(value, bool):
                self.errors.append(f"Expected boolean, got {type(value)}")
                return False
        elif expected_type == 'array':
            if not isinstance(value, list):
                self.errors.append(f"Expected array, got {type(value)}")
                return False
        elif expected_type == 'object':
            if not isinstance(value, dict):
                self.errors.append(f"Expected object, got {type(value)}")
                return False
        elif expected_type == 'null':
            if value is not None:
                self.errors.append(f"Expected null, got {type(value)}")
                return False
        
        return True
    
    def get_errors(self) -> List[str]:
        return self.errors.copy()

