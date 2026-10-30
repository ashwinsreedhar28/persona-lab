"""Preserve Transformers input checking across its decorator API variants."""
from functools import wraps
import inspect


def model_input_decorator(checker):
    """Kimi uses @check_model_inputs; 4.57.2 exposes @check_model_inputs().

    Adapt the call syntax rather than removing validation/output handling or changing the model's forward.
    """
    if getattr(checker,'_abliteration_compatible',False):
        return checker
    parameters=inspect.signature(checker).parameters
    if 'func' in parameters:
        return checker
    if 'tie_last_hidden_states' not in parameters:
        raise RuntimeError('Unsupported Transformers check_model_inputs API')
    @wraps(checker)
    def compatible(func=None,*,tie_last_hidden_states=True):
        if callable(func):
            return checker(tie_last_hidden_states=tie_last_hidden_states)(func)
        if func is not None:
            if not isinstance(func,bool):
                raise TypeError('Unexpected decorator argument')
            tie_last_hidden_states=func
        return checker(tie_last_hidden_states=tie_last_hidden_states)
    compatible._abliteration_compatible=True
    return compatible
