"""Value validation module for various data types and structures."""
from typing import Dict, Any, Optional, Union, List, Callable, Tuple, Type
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

@dataclass
class ValidationRule:
    """Represents a validation rule with a validator method and its arguments."""
    validator_method: str
    args: List[Any]
    kwargs: Dict[str, Any]

@dataclass
class NumericConstraints:
    """Constraints for numeric value validation."""
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    exclusive_minimum: Optional[Union[int, float]] = None
    exclusive_maximum: Optional[Union[int, float]] = None
    multiple_of: Optional[Union[int, float]] = None

class ValueValidator:
    """
    A validator for checking various types of values against constraints.
    
    Example usage:
        ```python
        validator = ValueValidator()
        
        # Validate a number
        constraints = NumericConstraints(minimum=0, maximum=100)
        is_valid, errors = validator.validate_numeric(42, constraints)
        
        # Validate a string
        is_valid, errors = validator.validate_string("test", min_length=2, max_length=10)
        
        # Validate nested data
        data = {
            "age": 25,
            "name": "John",
            "scores": [85, 90, 95]
        }
        validation_map = {
            "age": ValidationRule("validate_numeric", 
                [NumericConstraints(minimum=0, maximum=150)], {}),
            "name": ValidationRule("validate_string", 
                [], {"min_length": 2, "max_length": 50}),
            "scores": ValidationRule("validate_array", 
                [], {"min_items": 1, "max_items": 10})
        }
        is_valid, errors = validator.validate_nested(data, validation_map)
        
        # Example of composite validation
        rules = [
            ValidationRule("validate_type", [str], {"type_name": "string"}),
            ValidationRule("validate_string", [], {"min_length": 2, "max_length": 10}),
            ValidationRule("validate_with_custom_function", 
                [lambda x: x.isalpha()], {})
        ]
        is_valid, errors = validator.validate_composite("test", rules)
        ```
        
        # Example of validating elements in a list
        numbers = [1, 2, 3, -4, 5]
        num_validator = ValidationRule(
            "validate_numeric",
            [NumericConstraints(minimum=0)],
            {}
        )
        is_valid, errors = validator.validate_elements(numbers, num_validator)
        # Will report error for -4
        ```
    """
    
    def validate_composite(
        self,
        value: Any,
        rules: List[ValidationRule]
    ) -> Tuple[bool, List[str]]:
        """
        Apply multiple validation rules to a value.

        Args:
            value: The value to validate
            rules: List of ValidationRule objects defining the validation rules

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors: List[str] = []
        
        for rule in rules:
            validator = getattr(self, rule.validator_method, None)
            if validator is None:
                all_errors.append(f"Unknown validator method: {rule.validator_method}")
                continue
            
            is_valid, errors = validator(value, *rule.args, **rule.kwargs)
            if not is_valid:
                all_errors.extend(errors)
        
        return (len(all_errors) == 0, all_errors)

    def validate_nested(
        self,
        value: Dict[str, Any],
        validation_map: Dict[str, Union[ValidationRule, List[ValidationRule]]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a nested dictionary structure with different validation rules for each field.

        Args:
            value: The dictionary to validate
            validation_map: Mapping of field names to their validation rules

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors: List[str] = []
        
        for field_name, rules in validation_map.items():
            if field_name not in value:
                all_errors.append(f"Missing field: {field_name}")
                continue
            
            field_value = value[field_name]
            if isinstance(rules, list):
                is_valid, errors = self.validate_composite(field_value, rules)
            else:
                validator = getattr(self, rules.validator_method, None)
                if validator is None:
                    all_errors.append(f"Unknown validator method for field {field_name}: {rules.validator_method}")
                    continue
                is_valid, errors = validator(field_value, *rules.args, **rules.kwargs)
            
            if not is_valid:
                all_errors.extend([f"{field_name}: {error}" for error in errors])
        
        return (len(all_errors) == 0, all_errors)
    def validate_numeric(
        self,
        value: Union[int, float],
        constraints: NumericConstraints
    ) -> Tuple[bool, List[str]]:
        """
        Validate a numeric value against constraints.

        Args:
            value: The numeric value to validate
            constraints: NumericConstraints object containing validation rules

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        if constraints.minimum is not None and value < constraints.minimum:
            errors.append(f"Value {value} is less than minimum {constraints.minimum}")
        
        if constraints.maximum is not None and value > constraints.maximum:
            errors.append(f"Value {value} is greater than maximum {constraints.maximum}")
        
        if constraints.exclusive_minimum is not None and value <= constraints.exclusive_minimum:
            errors.append(f"Value {value} is not greater than exclusive minimum {constraints.exclusive_minimum}")
        
        if constraints.exclusive_maximum is not None and value >= constraints.exclusive_maximum:
            errors.append(f"Value {value} is not less than exclusive maximum {constraints.exclusive_maximum}")
        
        if constraints.multiple_of is not None:
            # Use Decimal for precise division checking
            if Decimal(str(value)) % Decimal(str(constraints.multiple_of)) != 0:
                errors.append(f"Value {value} is not a multiple of {constraints.multiple_of}")
        
        return (len(errors) == 0, errors)

    def validate_string(
        self,
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a string value against length and pattern constraints.

        Args:
            value: The string to validate
            min_length: Optional minimum length
            max_length: Optional maximum length
            pattern: Optional regex pattern to match

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        if min_length is not None and len(value) < min_length:
            errors.append(f"String length {len(value)} is less than minimum length {min_length}")
        
        if max_length is not None and len(value) > max_length:
            errors.append(f"String length {len(value)} is greater than maximum length {max_length}")
        
        if pattern is not None:
            import re
            if not re.match(pattern, value):
                errors.append(f"String does not match pattern {pattern}")
        
        return (len(errors) == 0, errors)

    def validate_array(
        self,
        value: List[Any],
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
        unique_items: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate an array/list against length and uniqueness constraints.

        Args:
            value: The list to validate
            min_items: Optional minimum number of items
            max_items: Optional maximum number of items
            unique_items: Whether items must be unique

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        if min_items is not None and len(value) < min_items:
            errors.append(f"Array length {len(value)} is less than minimum items {min_items}")
        
        if max_items is not None and len(value) > max_items:
            errors.append(f"Array length {len(value)} is greater than maximum items {max_items}")
        
        if unique_items and len(value) != len(set(str(x) for x in value)):
            errors.append("Array contains duplicate items")
        
        return (len(errors) == 0, errors)

    def validate_dict(
        self,
        value: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        allow_extra_keys: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a dictionary against key constraints.

        Args:
            value: The dictionary to validate
            required_keys: List of required keys
            optional_keys: List of optional keys
            allow_extra_keys: Whether to allow keys not in required or optional lists

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        # Check required keys
        if required_keys is not None:
            for key in required_keys:
                if key not in value:
                    errors.append(f"Missing required key: {key}")
        
        # Check for unknown keys
        if not allow_extra_keys and optional_keys is not None:
            allowed_keys = set(required_keys or []) | set(optional_keys)
            extra_keys = set(value.keys()) - allowed_keys
            if extra_keys:
                errors.append(f"Unknown keys found: {', '.join(extra_keys)}")
        
        return (len(errors) == 0, errors)

    def validate_with_custom_function(
        self,
        value: Any,
        validation_func: Callable[[Any], Union[bool, Tuple[bool, str]]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a value using a custom validation function.

        Args:
            value: The value to validate
            validation_func: Function that returns bool or Tuple[bool, str]

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        result = validation_func(value)
        if isinstance(result, tuple):
            is_valid, error_message = result
            if not is_valid:
                errors.append(error_message)
        elif not result:
            errors.append("Value failed custom validation")
        
        return (len(errors) == 0, errors)

    def validate_enum(self, value: Any, enum_class: Type[Enum]) -> Tuple[bool, List[str]]:
        """
        Validate that a value is a valid enum member.

        Args:
            value: The value to validate
            enum_class: The Enum class to check against

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        try:
            enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            errors.append(f"Value {value} is not a valid {enum_class.__name__}. Valid values are: {valid_values}")
        
        return (len(errors) == 0, errors)

    def validate_type(
        self,
        value: Any,
        expected_type: Union[Type, Tuple[Type, ...]],
        type_name: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a value is of the expected type.

        Args:
            value: The value to validate
            expected_type: Type or tuple of types to check against
            type_name: Optional friendly name for the type in error messages

        Returns:
            Tuple of (is_valid, list_of_errors)
            
        Example:
            ```python
            validator = ValueValidator()
            is_valid, errors = validator.validate_type(42, int)
            is_valid, errors = validator.validate_type("test", (str, bytes), "string")
            ```
        """
        errors: List[str] = []
        
        if not isinstance(value, expected_type):
            type_desc = type_name or (
                expected_type.__name__ if isinstance(expected_type, type)
                else ' or '.join(t.__name__ for t in expected_type)
            )
            errors.append(
                f"Expected {type_desc}, got {type(value).__name__}"
            )
        
        return (len(errors) == 0, errors)
        
    def validate_elements(
        self,
        value: Iterable[Any],
        element_validator: Union[ValidationRule, List[ValidationRule]],
        stop_on_first_error: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate each element in an iterable using the specified validation rules.

        Args:
            value: The iterable containing elements to validate
            element_validator: Validation rule(s) to apply to each element
            stop_on_first_error: Whether to stop validation after first error

        Returns:
            Tuple of (is_valid, list_of_errors)
            
        Example:
            ```python
            # Validate list of numbers
            numbers = [1, 2, 3, 4, 5]
            num_rule = ValidationRule(
                "validate_numeric",
                [NumericConstraints(minimum=0, maximum=10)],
                {}
            )
            is_valid, errors = validator.validate_elements(numbers, num_rule)

            # Validate list of strings with multiple rules
            strings = ["hello", "world", "python"]
            string_rules = [
                ValidationRule("validate_type", [str], {}),
                ValidationRule("validate_string", [], {"min_length": 3})
            ]
            is_valid, errors = validator.validate_elements(strings, string_rules)
            ```
        """
        all_errors: List[str] = []
        
        try:
            iterator = iter(value)
        except TypeError:
            return False, ["Value is not iterable"]
        
        for i, element in enumerate(iterator):
            if isinstance(element_validator, list):
                is_valid, errors = self.validate_composite(element, element_validator)
            else:
                validator = getattr(self, element_validator.validator_method, None)
                if validator is None:
                    all_errors.append(f"Unknown validator method: {element_validator.validator_method}")
                    if stop_on_first_error:
                        break
                    continue
                
                is_valid, errors = validator(element, *element_validator.args, **element_validator.kwargs)
            
            if not is_valid:
                indexed_errors = [f"Element {i}: {error}" for error in errors]
                all_errors.extend(indexed_errors)
                if stop_on_first_error:
                    break
        
        return (len(all_errors) == 0, all_errors)
