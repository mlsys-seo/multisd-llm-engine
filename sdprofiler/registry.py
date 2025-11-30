from typing import Any, Dict, Optional, Union, Type, get_type_hints
from dataclasses import dataclass, fields, is_dataclass
import inspect


DEBUG = False

class RegistryPath(object):
    """
    Registry path representation that can be used as dictionary key.
    Example paths:
        val.model.name
        fn.model.forward
        cls.model.tokenizer
        
    Attributes:
        raw (str): Raw string representation of the path
        parts (List[str]): Path components split by dots
        prefix (str): First part of the path (val/fn/cls)
        name (str): Remaining path after prefix
    """
    VALID_PREFIXES = {'val', 'fn', 'cls'}
    
    def __init__(
            self,
            current: str
        ):

        if not isinstance(current, str):
            raise TypeError(f"Path must be a string, got {type(current)}")
        
        self.raw = current
        self.parts = current.split('.')
        
        # Validate prefix on initial creation
        if len(self.parts) == 1:
            if self.parts[0] not in RegistryPath.VALID_PREFIXES:
                raise ValueError(f"Invalid prefix: {current}. Must be one of {RegistryPath.VALID_PREFIXES}")
        
        self.prefix = self.parts[0]
        self.name = '.'.join(self.parts[1:]) if len(self.parts) > 1 else ''
        
        # Validate prefix immediately
        if self.prefix not in RegistryPath.VALID_PREFIXES:
            raise ValueError(f"Invalid prefix: {self.prefix}. Must be one of {RegistryPath.VALID_PREFIXES}")
    
    # TODO: RegistryPath can't be pickled because of this
    # def __getattr__(self, name: str):
    #     if not name.isidentifier():
    #         raise ValueError(f"Invalid path component: {name}. Must be a valid Python identifier")
    #         
    #     if self.raw:
    #         new_path = f"{self.raw}.{name}" if self.raw else name
    #         return RegistryPath(new_path)
    #     else:
    #         return super().__getattr__(name)
    
    def __str__(self) -> str:
        return self.raw
    
    def __repr__(self) -> str:
        return f"RegistryPath({self.raw!r})"
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegistryPath):
            return NotImplemented
        return self.raw == other.raw
        
    def __hash__(self) -> int:
        return hash(self.raw)
    
    def is_valid(self) -> bool:
        if len(self.parts) < 2:
            return False
            
        if self.prefix not in RegistryPath.VALID_PREFIXES:
            return False
            
        return all(part.isidentifier() for part in self.parts)
    
    def parent(self) -> Optional['RegistryPath']:
        """Returns the parent path or None if this is a root path."""
        if len(self.parts) <= 1:
            return None
        return RegistryPath('.'.join(self.parts[:-1]))
    
    def leaf(self) -> str:
        """Returns the last component of the path."""
        return self.parts[-1] if self.parts else ''


@dataclass
class SchemaItem:
    """
    Schema definition for registry items.
    Includes type information, whether the item is required, and its default value.
    """
    type: Type
    required: bool = True
    default: Any = None


class RegistryNode:
    """
    Stores a value and its metadata in the registry.
    Supports callable values (functions and classes) through __call__.
    """
    def __init__(self, path: RegistryPath):
        self.path = path
        self.value: Any = None
        self.source_location: str = ""

    def __call__(self, *args, **kwargs):
        if callable(self.value):
            return self.value(*args, **kwargs)
        raise TypeError(f"Node {self.path} is not callable")


class EngineRegistry:
    """
    Schema-based registry that manages values, functions, and classes.
    Supports automatic registration of dataclass-based configs.
    """
    def __init__(self, *configs: dataclass):
        self._schema: Dict[RegistryPath, SchemaItem] = {}
        self._nodes: Dict[RegistryPath, RegistryNode] = {}
        
        if configs:
            self._register_configs(*configs)
    
    def _register_configs(self, *configs: dataclass) -> None:
        """
        Creates schema and registers default values from a dataclass instance.
        Paths are generated as {type}.{config_class_name_without_Config}.{field_name}
        Example: ModelConfig -> val.model.field_name
        """
        for config in configs:
            if not is_dataclass(config):
                raise ValueError(f"Config must be a dataclass instance, got {type(config)}")

        schema = {}
        for config in configs:
            config_cls = type(config)
            base_name = config_cls.__name__.removesuffix('Config').lower()
            
            if DEBUG:
                print(f"base_name: {base_name}")
                print(f"hasattr(config, 'type'): {hasattr(config, 'type')}")
                if hasattr(config, 'type'):
                    print(f"config.type: {config.type}")


            if hasattr(config, 'type') and config.type is not None:
                base_name = f"{config.type}_{base_name}"
            
            type_hints = get_type_hints(config_cls)
            for field in fields(config_cls):
                field_type = type_hints[field.name]
                
                if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                    field_type = field_type.__args__[0]
                
                prefix = self._get_prefix(field_type)

                path = f"{prefix}.{base_name}.{field.name}"
                schema[path] = {
                    "type": field_type,
                    "required": field.default is not None and field.default_factory is not None,
                    "default": getattr(config, field.name)
                }

        # Register schema
        self.define_schema(schema)
        
        # Register values
        for path_str, spec in schema.items():
            path = RegistryPath(path_str)
            self.register(path, spec["default"])

        self._check_dependencies()
            

    def _get_prefix(self, field_type):
        if isinstance(field_type, type) and field_type == type:
            prefix = "cls"
        elif field_type == callable:
            prefix = "fn"
        else:
            prefix = "val"
        return prefix


    def _print_grouped_entries(self, title: str, entries: dict, format_display_fn):
        """Helper function to print grouped registry entries with consistent formatting.
        
        Args:
            title (str): Title for the output section
            entries (dict): Dictionary of entries grouped by prefix
            format_display_fn: Function that takes an entry and returns its display string
        """
        max_length = max((len(str(path)) for path in entries.keys()), default=0)
        print("\n", f" {title} ".center(max_length + 28, '='))
        
        # Group
        grouped = {'cls': [], 'fn': [], 'val': []}
        for path, entry in entries.items():
            grouped[path.prefix].append((path, entry))
        
        # Print
        for prefix in ['cls', 'fn', 'val']:
            if grouped[prefix]:
                print("\n", f" {prefix} ".center(max_length + 14, '-'))
                for path, entry in sorted(grouped[prefix], key=lambda x: str(x[0])):
                    dots = '.' * (max_length - len(str(path)) + 3)
                    display = format_display_fn(entry)
                    print(f"{path} {dots} {display}")

        print("\n", f"".center(max_length + 28, '='))


    def print_schema(self):
        def format_schema_display(spec):
            required = "required" if spec.required else "optional"
            default = f", default={spec.default}" if spec.default is not None else ""
            return f"{spec.type.__name__} ({required}{default})"
        
        self._print_grouped_entries("Schema", self._schema, format_schema_display)


    def print_nodes(self):
        def format_node_display(node):
            if node.path.prefix in ('cls', 'fn'):
                return node.value.__name__ if node.value else 'None'
            return str(node.value)
        
        self._print_grouped_entries("Nodes", self._nodes, format_node_display)


    def _get_source_location(self, value: Any) -> str:
        """Get source location of the value. For functions and classes, returns their definition location."""
        if inspect.isfunction(value) or inspect.isclass(value):
            return f"{inspect.getfile(value)}:{inspect.getsourcelines(value)[1]}"
        
        # For regular values, return registration location
        frame = inspect.currentframe().f_back.f_back  # Skip _get_source_location and register method
        return f"{frame.f_code.co_filename}:{frame.f_lineno}"


    def define_schema(self, schema: Dict[str, Dict[str, Any]]) -> None:
        """Defines the registry's schema with types and requirements"""
        _schema: Dict[RegistryPath, SchemaItem] = {}
        for path_str, spec in schema.items():
            path = RegistryPath(path_str)
            _schema[path] = SchemaItem(**spec)
        if self._schema is not None:
            self._schema.update(_schema)
        else:
            self._schema = _schema

    def _check_dependencies(self) -> None:
        global_world_size = self.get(RegistryPath('val.model.world_size'))
        num_models_for_speculation = 1
        min_vocab_size = self.get(RegistryPath('val.model.vocab_size'))
        self.define_schema({
            'val.model.global_ranks': {
                'type': list, 'required': True, 'default': []}
        })
        self.register(RegistryPath('val.model.global_ranks'), [i for i in range(global_world_size)])

        if any('draft_model' in str(path) for path in self._schema.keys()):
            if self.get(RegistryPath('val.engine.smart_spec_enabled')):
                from .worker import SmartSpecWorker
                from .request import SmartSpecRequest
                self.register(RegistryPath('cls.engine.worker_class'), SmartSpecWorker)
                self.register(RegistryPath('cls.engine.request_class'), SmartSpecRequest)
            elif self.get(RegistryPath('val.engine.svip_enabled')):
                from .worker import SVIPWorker
                from .request import SVIPRequest
                self.register(RegistryPath('cls.engine.worker_class'), SVIPWorker)
                self.register(RegistryPath('cls.engine.request_class'), SVIPRequest)
            elif self.get(RegistryPath('val.engine.eagle3_enabled')):
                from .worker import Eagle3Worker
                from .request import Eagle3Request
                self.register(RegistryPath('cls.engine.worker_class'), Eagle3Worker)
                self.register(RegistryPath('cls.engine.request_class'), Eagle3Request)
            
            elif "layerskip" in self._schema[RegistryPath('val.model.hf_config')].default._name_or_path.lower():
                from .worker import LayerSkipWorker
                from .request import LayerSkipRequest
                self.register(RegistryPath('cls.engine.worker_class'), LayerSkipWorker)
                self.register(RegistryPath('cls.engine.request_class'), LayerSkipRequest)

            else:
                from .worker import SpeculativeWorker
                from .request import SpeculativeRequest
                self.register(RegistryPath('cls.engine.worker_class'), SpeculativeWorker)
                self.register(RegistryPath('cls.engine.request_class'), SpeculativeRequest)
            
            
            local_world_size = self.get(RegistryPath('val.draft_model.world_size'))
            self.define_schema({
                'val.draft_model.global_ranks': {
                    'type': list, 'required': True, 'default': []}
            })
            self.register(RegistryPath('val.draft_model.global_ranks'), [i+global_world_size for i in range(local_world_size)])
            global_world_size = global_world_size + local_world_size
            num_models_for_speculation += 1

            min_vocab_size = min(min_vocab_size, self.get(RegistryPath('val.draft_model.vocab_size')))

            if self.get(RegistryPath('val.engine.num_draft_steps')) is None:
                raise KeyError("The key 'val.engine.num_draft_steps' is not defined in the registry schema.")

        if any('verify_model' in str(path) for path in self._schema.keys()):
            if self.get(RegistryPath('val.engine.eagle3_enabled')):
                from .worker import Eagle3SVWorker
                from .request import Eagle3SVRequest
                self.register(RegistryPath('cls.engine.worker_class'), Eagle3SVWorker)
                self.register(RegistryPath('cls.engine.request_class'), Eagle3SVRequest)

            elif "layerskip" in self._schema[RegistryPath('val.model.hf_config')].default._name_or_path.lower():
                from .worker import HierarchicalLayerSkipWorker
                from .request import HierarchicalLayerSkipRequest
                self.register(RegistryPath('cls.engine.worker_class'), HierarchicalLayerSkipWorker)
                self.register(RegistryPath('cls.engine.request_class'), HierarchicalLayerSkipRequest)

            else:
                from .worker import HierarchicalSpeculativeWorker
                from .request import HierarchicalSpeculativeRequest
                self.register(RegistryPath('cls.engine.worker_class'), HierarchicalSpeculativeWorker)
                self.register(RegistryPath('cls.engine.request_class'), HierarchicalSpeculativeRequest)
            
            local_world_size = self.get(RegistryPath('val.verify_model.world_size'))
            self.define_schema({
                'val.verify_model.global_ranks': {
                    'type': list, 'required': True, 'default': []}
            })
            self.register(RegistryPath('val.verify_model.global_ranks'), [i+global_world_size for i in range(local_world_size)])
            global_world_size = global_world_size + local_world_size
            num_models_for_speculation += 1

            min_vocab_size = min(min_vocab_size, self.get(RegistryPath('val.verify_model.vocab_size')))

            if self.get(RegistryPath('val.engine.num_draft_steps')) is None:
                raise KeyError("The key 'val.engine.num_draft_steps' is not defined in the registry schema.")

        self.define_schema({
            'val.engine.global_world_size': {
                'type': int, 'required': True, 'default': 0},
            'val.engine.vocab_size': {
                'type': int, 'required': True, 'default': 0}
        })

        self.register(RegistryPath('val.engine.vocab_size'), min_vocab_size)
        self.register(RegistryPath('val.engine.global_world_size'), global_world_size)
        # if global_world_size > num_models_for_speculation:
        #     if not self.get(RegistryPath('val.engine.use_ray_worker')):
        #         self.register(RegistryPath('val.engine.use_ray_executor'), True)
        #     else:
        #         self.register(RegistryPath('val.engine.use_ray_executor'), False)

        if self.get(RegistryPath('val.engine.use_ray_worker')):
            self.register(RegistryPath('val.engine.global_world_size'), self.get(RegistryPath('val.model.world_size')))
            if self.get(RegistryPath('val.engine.num_pipelining_steps')) > 1:
                devide = self.get(RegistryPath('val.engine.global_world_size')) if self.get(RegistryPath('val.engine.use_data_parallel_draft')) else 1
                self.register(RegistryPath('val.engine.num_max_batch_requests_pipe'), self.get(RegistryPath('val.engine.num_max_batch_requests')) // devide)
                self.register(RegistryPath('val.engine.num_min_batch_requests_pipe'), (self.get(RegistryPath('val.engine.num_min_batch_requests')) // devide) if self.get(RegistryPath('val.engine.num_min_batch_requests')) is not None else None)
                self.register(RegistryPath('val.engine.use_ray_executor'), True)
            else:
                self.register(RegistryPath('val.engine.use_ray_executor'), False)
        else:
            self.register(RegistryPath('val.engine.global_world_size'), self.get(RegistryPath('val.model.world_size')))
            if self.get(RegistryPath('val.model.world_size')) > 1:
                self.register(RegistryPath('val.engine.num_max_batch_requests_pipe'), self.get(RegistryPath('val.engine.num_max_batch_requests')))
                self.register(RegistryPath('val.engine.num_min_batch_requests_pipe'), (self.get(RegistryPath('val.engine.num_min_batch_requests'))) if self.get(RegistryPath('val.engine.num_min_batch_requests')) is not None else None)
                self.register(RegistryPath('val.engine.use_ray_executor'), True)

        # CUDA Graph
        max_local_num_requests = self.get(RegistryPath('val.engine.num_max_batch_requests'))
        if self.get(RegistryPath('val.engine.use_data_parallel_draft')):
            max_local_num_requests //= self.get(RegistryPath('val.model.world_size'))

        cuda_graph_target_batch_sizes =[2**i for i in range(0, max_local_num_requests.bit_length()) if 2**i <= max_local_num_requests]
        if max_local_num_requests not in cuda_graph_target_batch_sizes:
            cuda_graph_target_batch_sizes.append(max_local_num_requests)
        self.register(RegistryPath('val.engine.cuda_graph_target_batch_sizes'), cuda_graph_target_batch_sizes)


    def _validate_value(self, path: RegistryPath, value: Any) -> None:
        """
        Validates that a value matches its schema definition and path prefix rules.
        - cls.* paths must contain class types
        - fn.* paths must contain callable objects
        - val.* paths must contain values matching their schema type
        """
        if path not in self._schema:
            raise KeyError(f"Path {path} not defined in schema")
            
        if value is None:
            return

        # Handle class type registration (cls.*)
        if path.prefix == 'cls':
            if not isinstance(value, type):
                raise TypeError(f"Expected a class for {path}, got {type(value)}")
            return

        # Handle function registration (fn.*)
        if path.prefix == 'fn':
            if not callable(value):
                raise TypeError(f"Expected a callable for {path}, got {type(value)}")
            return

        # Handle value registration (val.*)
        if path.prefix == 'val':
            schema_item = self._schema[path]
            if schema_item.type is not Any:
                # Get the origin type for generics (e.g., List[str] -> list)
                origin_type = getattr(schema_item.type, "__origin__", schema_item.type)
                
                # Special handling for Union types (including Optional)
                if origin_type is Union:
                    allowed_types = schema_item.type.__args__
                    if not any(isinstance(value, t) for t in allowed_types):
                        raise TypeError(f"Expected one of {allowed_types} for {path}, got {type(value)}")
                # Handle other generic types
                elif origin_type is not None:
                    if not isinstance(value, origin_type):
                        raise TypeError(f"Expected {origin_type} for {path}, got {type(value)}")
                # Handle non-generic types
                else:
                    if not isinstance(value, schema_item.type):
                        raise TypeError(f"Expected {schema_item.type} for {path}, got {type(value)}")
            return


    def register(self, path: Union[str, RegistryPath], value: Any) -> None:
        """Registers a value, function, or class at the specified path"""
        if not isinstance(path, RegistryPath):
            path = RegistryPath(path)
        
        if not path.is_valid():
            raise ValueError(f"Invalid path: {path}")
        
        self._validate_value(path, value)
        
        if path not in self._nodes:
            self._nodes[path] = RegistryNode(path)
        
        node = self._nodes[path]
        node.value = value
        node.source_location = self._get_source_location(value)


    def get(self, path: Union[str, RegistryPath]) -> Any:
        """
        Retrieves a registered value.
        Returns the default value if available and no value is registered.
        """
        if not isinstance(path, RegistryPath):
            path = RegistryPath(path)

        if path not in self._nodes:
            if path in self._schema:
                default = self._schema[path].default
                if default is not None:
                    return default
            return None
        return self._nodes[path].value


    def get_source_location(self, path: Union[str, RegistryPath]) -> str:
        """Returns the source location of the registered value"""
        if not isinstance(path, RegistryPath):
            path = RegistryPath(path)
        return self._nodes[path].source_location


    def get_schema(self) -> Dict[RegistryPath, SchemaItem]:
        """Returns the registry's schema"""
        return self._schema


    def __getattr__(self, name: str) -> RegistryPath:
        """Enables attribute-style access to registry paths"""
        if name in {'val', 'fn', 'cls'}:
            return RegistryPath(f"{name}")
        raise AttributeError(f"Invalid registry access: {name}")


if __name__ == "__main__":
    import torch
    from .config import ModelConfig

    class Tokenizer:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def tokenize(self, text: str) -> list:
            return text.split()


    model_config = ModelConfig(
        model_name_or_path="gpt2",
        device="cuda"
    )

    print("Initializing registry with model config")
    registry = EngineRegistry(model_config)
    print(registry.get(registry.val.model.model_name_or_path))
    print(registry.get('val.model.device'))

    print("\n------------------------------------------------")
    print("Defining additional schema")
    schema_dict = {
        "fn.model.generate": {"type": callable, "required": True},
        "cls.model.tokenizer": {"type": type, "required": True}
    }
    # Define additional schema for functions and classes
    registry.define_schema(schema_dict)
    print(registry.get_schema())

    print("\n------------------------------------------------")
    print(f"Class registration")
    registry.register(registry.cls.model.tokenizer, Tokenizer)  # OK - registering the class itself
    print(f"  try registering instance")
    try:
        tokenizer_instance = Tokenizer(1000)
        registry.register(registry.cls.model.tokenizer, tokenizer_instance)  # Error - can't register instance to cls.*
    except TypeError as e:
        print(f"Expected error when registering instance to cls.*: {e}")
    print(f"  try registering class")
    registry.register(registry.cls.model.tokenizer, Tokenizer)  # OK - registering the class itself
    print(registry.get(registry.cls.model.tokenizer))


    print("\n------------------------------------------------")
    print(f"Function registration")
    def generate_text(prompt: str, max_length: int = 100) -> str:
        return f"Generated text for: {prompt}"
    registry.register(registry.fn.model.generate, generate_text)  # OK - function registration
    print(registry.get(registry.fn.model.generate))
    print(f"source location: {registry.get_source_location(registry.fn.model.generate)}")

    print("\n------------------------------------------------")
    print(f"Value registration")
    registry.define_schema({
        "val.model.batch_size": {"type": int, "required": True}
    })
    registry.register(registry.val.model.batch_size, 32)  # OK - value registration
    print(registry.get(registry.val.model.batch_size))
