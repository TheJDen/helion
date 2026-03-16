from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import pathlib
import tempfile
from typing import TYPE_CHECKING

from torch._functorch.aot_autograd import aot_module_simplified
import torch._functorch.config as functorch_config
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.decomposition import select_decomp_table
import torch.fx
from torch.fx.node import Node
from torch.fx.node import map_arg

from .. import exc

if TYPE_CHECKING:
    from ..runtime.kernel import Kernel


@dataclass
class MatmulPatternInfo:
    """Information about a detected matmul pattern in the forward kernel."""

    a_name: str  # left operand tensor name
    b_name: str  # right operand tensor name
    input_names: list[str]  # all input tensor names in order
    batched: bool  # True for bmm/baddbmm (3D), False for mm/addmm (2D)


@dataclass
class InputMapping:
    placeholder_name: str
    tensor_name: str
    fake_tensor: torch.Tensor | None


@dataclass
class OutputMapping:
    tensor_name: str
    fake_tensor: torch.Tensor | None


class GraphAnalyzer:
    """
    Analyzes forward Helion graph and extracts the pure computation subgraph.
    """

    def __init__(self, forward_graph: torch.fx.Graph) -> None:
        self.forward_graph = forward_graph

    def _get_tensor_name(self, host_tensor_node: Node) -> str:
        target = host_tensor_node.target
        assert callable(target) and getattr(target, "__name__", "") == "_host_tensor"
        name = host_tensor_node.args[0]
        assert isinstance(name, str)
        return name

    def extract_computation_graph(
        self,
    ) -> tuple[torch.fx.Graph, list[InputMapping], list[OutputMapping]]:
        """
        Extract computation subgraph.

        Returns:
            compute_graph: Pure PyTorch FX graph
            input_mappings: Load -> placeholder mappings
            output_mappings: Store -> output tensor mappings
        """
        compute_graph = torch.fx.Graph()
        node_map: dict[Node, Node] = {}
        input_mappings: list[InputMapping] = []

        # Track current value of each tensor (None = need placeholder for original)
        tensor_to_placeholder: dict[str, Node] = {}
        tensor_current_value: dict[str, Node] = {}

        # Process nodes in order to preserve load/store semantics
        for node in self.forward_graph.nodes:
            if node.op != "call_function":
                continue

            target = node.target
            assert callable(target)
            target_name = target.__name__

            if target_name == "load":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                fake_tensor = node.meta["val"]

                # Check if there's a prior store to this tensor
                if tensor_name in tensor_current_value:
                    # Load after store: use the stored value
                    stored_value_node = tensor_current_value[tensor_name]
                    node_map[node] = node_map[stored_value_node]
                elif tensor_name in tensor_to_placeholder:
                    # Another load of same tensor (before any store): reuse placeholder
                    node_map[node] = tensor_to_placeholder[tensor_name]
                else:
                    # First load of this tensor: create placeholder
                    ph_name = f"tile_{tensor_name}"
                    ph = compute_graph.placeholder(ph_name)
                    tensor_to_placeholder[tensor_name] = ph
                    node_map[node] = ph

                    input_mappings.append(
                        InputMapping(
                            placeholder_name=ph_name,
                            tensor_name=tensor_name,
                            fake_tensor=fake_tensor,
                        )
                    )

            elif target_name == "store":
                host_tensor_node = node.args[0]
                assert isinstance(host_tensor_node, Node)
                tensor_name = self._get_tensor_name(host_tensor_node)
                value_node = node.args[2]
                if isinstance(value_node, Node):
                    tensor_current_value[tensor_name] = value_node

            elif target_name == "_mask_to":
                # _mask_to(tensor, fill_value) pads masked tile elements for
                # reductions. The backward Helion kernel gets its own masking
                # from the Helion compiler, so we pass through as identity.
                input_node = node.args[0]
                if isinstance(input_node, Node) and input_node in node_map:
                    node_map[node] = node_map[input_node]

            elif target_name in ("_host_tensor", "_get_symnode"):
                # Helion infrastructure for tensor/symbol references.
                # Already consumed by load/store arg processing above.
                pass

            elif target_name == "_inductor_lowering_extra":
                # Helion internal: holds references to input nodes that
                # strip_unused_inputs moved out of the main op's args.
                # Not computation — just track it so we can restore args.
                pass

            elif target_name.startswith("_"):
                raise exc.AutodiffNotSupported(f"Helion internal op '{target_name}'")

            else:
                # Computation node: copy to computation graph
                args = node.args
                # Helion's strip_unused_inputs replaces args with None when
                # they reference a buffer already used by another arg. Restore
                # them from _extra_args (which points to the
                # _inductor_lowering_extra node holding the real inputs) or
                # from the first remaining Node arg.
                extra_args = node.kwargs.get("_extra_args")
                if extra_args is not None and isinstance(extra_args, (list, tuple)):
                    for ea in extra_args:
                        if not isinstance(ea, Node):
                            continue
                        # _inductor_lowering_extra([real_input_node, ...])
                        real_inputs = ea.args[0]
                        if isinstance(real_inputs, (list, tuple)):
                            for ri in real_inputs:
                                if isinstance(ri, Node):
                                    args = tuple(ri if a is None else a for a in args)
                                    break

                first_node_arg = next((a for a in args if isinstance(a, Node)), None)
                if first_node_arg is not None:
                    args = tuple(first_node_arg if a is None else a for a in args)

                new_args = map_arg(args, node_map.get)
                # Strip _extra_args (Helion-internal kwarg)
                clean_kwargs = {
                    k: v for k, v in node.kwargs.items() if k != "_extra_args"
                }
                new_kwargs = map_arg(clean_kwargs, node_map.get)
                target = node.target
                assert callable(target)
                new_node = compute_graph.call_function(target, new_args, new_kwargs)
                if node.meta:
                    # Only preserve "val" metadata; strip Helion-internal keys
                    new_node.meta = {k: v for k, v in node.meta.items() if k == "val"}
                node_map[node] = new_node

        input_tensor_names = set(tensor_to_placeholder.keys())
        output_mappings: list[OutputMapping] = []
        outputs = []
        for tensor_name, value_node in tensor_current_value.items():
            if tensor_name not in input_tensor_names:
                mapped_node = node_map[value_node]
                outputs.append(mapped_node)
                fake_tensor = mapped_node.meta.get("val") if mapped_node.meta else None
                output_mappings.append(
                    OutputMapping(
                        tensor_name=tensor_name,
                        fake_tensor=fake_tensor,
                    )
                )
        compute_graph.output(tuple(outputs))
        return compute_graph, input_mappings, output_mappings


def differentiate_graph(
    compute_graph: torch.fx.Graph,
    input_tensors: tuple[torch.Tensor, ...],
) -> torch.fx.Graph:
    """
    Differentiate computation graph using AOT Autograd with full recomputation.

    Returns:
        backward_graph: FX graph for backward computation
    """
    example_inputs = [
        torch.empty(t.shape, dtype=t.dtype, device=t.device, requires_grad=True)
        for t in input_tensors
    ]

    # Capture backward graph via compiler callback
    backward_graph: torch.fx.Graph | None = None

    def bw_compiler(
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ) -> torch.fx.GraphModule:
        nonlocal backward_graph
        backward_graph = gm.graph
        return gm

    with functorch_config.patch(activation_memory_budget=0):
        compiled = aot_module_simplified(
            torch.fx.GraphModule({}, compute_graph),
            example_inputs,
            fw_compiler=lambda gm, _: gm,  # type: ignore[arg-type]
            bw_compiler=bw_compiler,  # type: ignore[arg-type]
            decompositions=select_decomp_table(),
            partition_fn=min_cut_rematerialization_partition,
        )

        example_out = compiled(*example_inputs)
        if isinstance(example_out, (list, tuple)):
            loss = sum(o.sum() for o in example_out)
        else:
            loss = example_out.sum()
        assert isinstance(loss, torch.Tensor)
        loss.backward()

    assert backward_graph is not None
    return backward_graph


class FXToHelionConverter:
    """Converts backward FX graph to Helion kernel source code."""

    def __init__(
        self,
        backward_graph: torch.fx.Graph,
        input_mappings: list[InputMapping],
        input_tensors: tuple[torch.Tensor, ...],
        grad_out_shape: tuple[int, ...],
    ) -> None:
        self.backward_graph = backward_graph
        self.grad_input_order = [m.tensor_name for m in input_mappings]
        self.grad_out_shape = grad_out_shape

        # Map primal index (1-based from AOT Autograd) to tensor name
        self.primal_to_name = {
            i + 1: m.tensor_name for i, m in enumerate(input_mappings)
        }

        # Map tensor name to concrete shape from real input tensors
        self.tensor_shapes: dict[str, tuple[int, ...]] = {
            m.tensor_name: tuple(input_tensors[i].shape)
            for i, m in enumerate(input_mappings)
        }

    def convert(self) -> str:
        """Convert the backward FX graph to Helion kernel source code."""
        placeholders = []
        computations = []
        output_node = None

        for node in self.backward_graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
            elif node.op == "call_function":
                computations.append(node)
            elif node.op == "output":
                output_node = node

        # Generate code components
        input_params = ["grad_out", *self.grad_input_order]
        output_grad_names = [f"grad_{name}" for name in self.grad_input_order]
        computation_lines = self._generate_computation(computations, placeholders)
        output_assignments = self._generate_output_assignments(output_node)

        return self._build_source(
            input_params, output_grad_names, computation_lines, output_assignments
        )

    def _get_var_name(self, node_name: str) -> str:
        """Map backward graph node name to generated variable name."""
        if node_name.startswith("primals_"):
            idx = int(node_name.split("_")[1])
            return f"{self.primal_to_name[idx]}_tile"
        if node_name.startswith("tangents_"):
            return "grad_out_tile"
        return f"{node_name}_val"

    def _find_tensor_by_shape(self, shape: tuple[int, ...]) -> str | None:
        """Match a concrete shape against known tensor shapes."""
        for name, tensor_shape in self.tensor_shapes.items():
            if tensor_shape == shape:
                return name
        return None

    def _generate_shape_aware_code(
        self,
        op_name: str,
        node: Node,
        arg_vars: list[str],
        node_to_var: dict[str, str],
    ) -> str | None:
        """Generate code for shape-changing ops (expand, view, unsqueeze).

        Returns generated code string, or None if this op doesn't need special handling.
        """
        if op_name == "unsqueeze":
            # Convert unsqueeze to view to work around Helion's unsqueeze
            # lowering issue on 1D tiles. unsqueeze(tensor, dim) where
            # tensor is 1D becomes tensor.view(tensor.shape[0], 1).
            dim_arg = node.args[1] if len(node.args) > 1 else None
            if dim_arg is not None:
                input_node = node.args[0]
                assert isinstance(input_node, Node)
                input_var = node_to_var.get(input_node.name, arg_vars[0])
                # Build view shape: insert 1 at the unsqueeze dim
                # For the common case of unsqueeze(-1) on a 1D tensor:
                # view(shape[0], 1)
                input_meta = input_node.meta.get("val")
                if input_meta is not None:
                    ndim = input_meta.ndim
                    dim = dim_arg if dim_arg >= 0 else ndim + 1 + dim_arg
                    shape_parts = []
                    for i in range(ndim):
                        shape_parts.append(f"{input_var}.shape[{i}]")
                    shape_parts.insert(dim, "1")
                    return f"{input_var}.view({', '.join(shape_parts)})"
            return None

        if op_name == "expand":
            # expand(tensor, [M, N]) -> expand_as(matching_input_tile)
            shape_arg = node.args[1]
            if isinstance(shape_arg, (list, tuple)):
                match = self._find_tensor_by_shape(tuple(shape_arg))
                if match:
                    return f"{arg_vars[0]}.expand_as({match}_tile)"
            return None

        if op_name in ("view", "reshape"):
            # view(tensor, [M, 1]) -> view with dynamic shape
            shape_arg = node.args[1]
            if isinstance(shape_arg, (list, tuple)):
                shape_tuple = tuple(shape_arg)
                # Check if it matches a known tensor shape
                match = self._find_tensor_by_shape(shape_tuple)
                if match:
                    return f"{arg_vars[0]}.view({match}_tile.shape)"
                # Build dynamic view shape by matching dims against known shapes
                for name, tensor_shape in self.tensor_shapes.items():
                    if len(shape_tuple) != len(tensor_shape):
                        continue
                    dynamic_dims = []
                    for i, (s, t) in enumerate(
                        zip(shape_tuple, tensor_shape, strict=True)
                    ):
                        if s == t:
                            dynamic_dims.append(f"{name}_tile.shape[{i}]")
                        elif s == 1:
                            dynamic_dims.append("1")
                        else:
                            break
                    else:
                        shape_expr = ", ".join(dynamic_dims)
                        return f"{arg_vars[0]}.view({shape_expr})"
                # Check against grad_out shape
                grad_shape = self.grad_out_shape
                if len(shape_tuple) == len(grad_shape):
                    dynamic_dims = []
                    for i, (s, g) in enumerate(
                        zip(shape_tuple, grad_shape, strict=True)
                    ):
                        if s == g:
                            dynamic_dims.append(f"grad_out_tile.shape[{i}]")
                        elif s == 1:
                            dynamic_dims.append("1")
                        else:
                            break
                    else:
                        shape_expr = ", ".join(dynamic_dims)
                        return f"{arg_vars[0]}.view({shape_expr})"
            return None

        return None

    def _generate_computation(
        self, computations: list[Node], placeholders: list[Node]
    ) -> list[str]:
        """Generate Python code for each computation node.

        With full recomputation (activation_memory_budget=0), the backward graph
        contains all forward computation ops. Placeholders are only primals_
        (original inputs) and tangents_ (upstream gradients).
        """
        lines = []
        node_to_var: dict[str, str] = {}

        def process_arg(arg: object) -> str:
            if isinstance(arg, Node):
                return node_to_var[arg.name]
            if isinstance(arg, (list, tuple)):
                processed = [process_arg(item) for item in arg]
                return f"[{', '.join(processed)}]"
            return repr(arg)

        # Map all placeholders (primals_ and tangents_) to variable names
        for ph in placeholders:
            node_to_var[ph.name] = self._get_var_name(ph.name)

        # Generate code for each computation node
        for node in computations:
            target = node.target
            op_name = getattr(target, "_opname", None)

            # Skip identity ops - just alias to input variable
            if op_name in {"detach", "alias"}:
                if node.args:
                    input_node = node.args[0]
                    assert isinstance(input_node, Node)
                    node_to_var[node.name] = node_to_var[input_node.name]
                    continue

            # Skip scalar_tensor - will inline the literal value
            if op_name == "scalar_tensor":
                if node.args:
                    node_to_var[node.name] = repr(node.args[0])
                    continue

            result_var = f"{node.name}_val"
            node_to_var[node.name] = result_var

            arg_vars = [process_arg(arg) for arg in node.args]

            # Cast bool inputs to int32 before reductions to avoid Triton
            # type mismatch in loop-carried accumulators (bool -> int64).
            if op_name in ("sum", "mean", "prod") and node.args:
                input_node = node.args[0]
                if isinstance(input_node, Node):
                    input_val = input_node.meta.get("val")
                    if input_val is not None and input_val.dtype == torch.bool:
                        cast_var = f"{arg_vars[0]}.to(torch.int64)"
                        cast_name = f"{node.name}_cast"
                        lines.append(f"{cast_name} = {cast_var}")
                        arg_vars[0] = cast_name

            # Try shape-aware codegen for expand/view ops
            code = None
            if op_name is not None:
                code = self._generate_shape_aware_code(
                    op_name, node, arg_vars, node_to_var
                )

            if code is None:
                # Generate op code: torch function or tensor method
                if op_name is not None and arg_vars:
                    if hasattr(torch, op_name):
                        code = f"torch.{op_name}({', '.join(arg_vars)})"
                    else:
                        tensor = arg_vars[0]
                        method_args = ", ".join(arg_vars[1:])
                        code = f"{tensor}.{op_name}({method_args})"
                elif op_name is not None:
                    code = f"torch.{op_name}({', '.join(arg_vars)})"
                else:
                    code = f"{node.target}({', '.join(arg_vars)})"

            lines.append(f"{result_var} = {code}")

        return lines

    def _generate_output_assignments(
        self, output_node: Node | None
    ) -> list[tuple[str, str]]:
        """
        Map backward graph outputs to gradient variable assignments.
        The backward graph returns gradients in the same order as forward inputs
        """
        if output_node is None:
            return []

        # FX output node stores return values in args[0]
        output_args = output_node.args[0]
        if isinstance(output_args, (list, tuple)):
            output_args_list = list(output_args)
        else:
            output_args_list = [output_args]

        # Pair each output with its corresponding gradient name
        assignments = []
        for i, out_node in enumerate(output_args_list):
            grad_name = f"grad_{self.grad_input_order[i]}"
            assert isinstance(out_node, Node)
            var_name = self._get_var_name(out_node.name)
            assignments.append((grad_name, var_name))

        return assignments

    def _build_source(
        self,
        input_params: list[str],
        output_grad_names: list[str],
        computation_lines: list[str],
        output_assignments: list[tuple[str, str]],
    ) -> str:
        """Build the complete Helion kernel source code using AST."""

        # Determine iteration shape. For reductions, iterate over the
        # grad_out shape (non-reduced dims) so that loads of input tensors
        # use full slices on the reduced dims — matching the forward kernel's
        # tiling structure.
        iter_var = output_grad_names[0]
        iter_tensor_name = iter_var.replace("grad_", "")
        input_ndim = len(self.tensor_shapes[iter_tensor_name])
        grad_out_ndim = len(self.grad_out_shape)
        is_reduction = grad_out_ndim < input_ndim
        if is_reduction:
            # Iterate over grad_out shape so each tile sees full rows
            iter_var = "grad_out"
            iter_ndim = grad_out_ndim
        else:
            iter_ndim = input_ndim

        def parse_expr(code: str) -> ast.expr:
            return ast.parse(code, mode="eval").body

        def parse_stmt(code: str) -> ast.stmt:
            return ast.parse(code, mode="exec").body[0]

        # Build imports
        imports: list[ast.stmt] = [
            ast.Import(names=[ast.alias(name="torch", asname=None)]),
            ast.Import(names=[ast.alias(name="helion", asname=None)]),
            ast.ImportFrom(
                module="helion",
                names=[ast.alias(name="language", asname="hl")],
                level=0,
            ),
        ]

        # Build function parameters: (grad_out: torch.Tensor, x: torch.Tensor, ...)
        tensor_annotation = parse_expr("torch.Tensor")
        func_args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p, annotation=tensor_annotation) for p in input_params],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )

        # Return type: torch.Tensor or tuple[torch.Tensor, ...]
        multi_output = len(output_grad_names) > 1
        return_annotation = parse_expr(
            "tuple[torch.Tensor, ...]" if multi_output else "torch.Tensor"
        )

        # Function body before loop: grad_x = torch.empty_like(x)
        body: list[ast.stmt] = [
            parse_stmt(f"{g} = torch.empty_like({g.replace('grad_', '')})")
            for g in output_grad_names
        ]

        # Loop body: loads → computation → stores
        loop_body: list[ast.stmt] = []

        # Load statements
        for p in input_params:
            tensor_ndim = (
                len(self.grad_out_shape)
                if p == "grad_out"
                else len(self.tensor_shapes[p])
            )
            if tensor_ndim == iter_ndim:
                load_expr = f"{p}[tile]"
            elif tensor_ndim > iter_ndim:
                # Input has more dims than iteration (reduction case):
                # use tile for iterated dims, : for reduced dims
                indices = (
                    ["tile"]
                    if iter_ndim == 1
                    else [f"tile[{i}]" for i in range(iter_ndim)]
                )
                indices.extend(":" for _ in range(tensor_ndim - iter_ndim))
                load_expr = f"{p}[{', '.join(indices)}]"
            else:
                # Fewer dims than iteration (e.g., grad_out in elementwise)
                indices = ", ".join(f"tile[{i}]" for i in range(tensor_ndim))
                load_expr = f"{p}[{indices}]"
            loop_body.append(parse_stmt(f"{p}_tile = {load_expr}"))

        # Computation statements
        for line in computation_lines:
            loop_body.append(parse_stmt(line))

        # Store statements: grad_x[tile, :] = value (for reductions)
        for grad_name, var_name in output_assignments:
            store_tensor_name = grad_name.replace("grad_", "")
            store_ndim = len(self.tensor_shapes[store_tensor_name])
            if store_ndim > iter_ndim:
                indices = (
                    ["tile"]
                    if iter_ndim == 1
                    else [f"tile[{i}]" for i in range(iter_ndim)]
                )
                indices.extend(":" for _ in range(store_ndim - iter_ndim))
                store_idx = ", ".join(indices)
                loop_body.append(parse_stmt(f"{grad_name}[{store_idx}] = {var_name}"))
            else:
                loop_body.append(parse_stmt(f"{grad_name}[tile] = {var_name}"))

        # For loop: for tile in hl.tile(iter_var.shape):
        body.append(
            ast.For(
                target=ast.Name(id="tile", ctx=ast.Store()),
                iter=parse_expr(f"hl.tile({iter_var}.shape)"),
                body=loop_body,
                orelse=[],
            )
        )

        # Return statement
        if multi_output:
            return_value = ast.Tuple(
                elts=[ast.Name(id=g, ctx=ast.Load()) for g in output_grad_names],
                ctx=ast.Load(),
            )
        else:
            return_value = ast.Name(id=output_grad_names[0], ctx=ast.Load())
        body.append(ast.Return(value=return_value))

        # Function definition with @helion.kernel() decorator
        func_def = ast.FunctionDef(
            name="backward_kernel",
            args=func_args,
            body=body,
            decorator_list=[parse_expr("helion.kernel()")],
            returns=return_annotation,
        )

        module = ast.Module(body=[*imports, func_def], type_ignores=[])
        ast.fix_missing_locations(module)

        source = ast.unparse(module)
        header = '"""\nAuto-generated Helion backward kernel.\n"""\n\n'
        return header + source


def _get_host_tensor_name(node: Node) -> str | None:
    """Trace a node back to a _host_tensor call and return the tensor name."""
    if node.op != "call_function":
        return None
    if getattr(node.target, "__name__", "") == "_host_tensor":
        name = node.args[0]
        return name if isinstance(name, str) else None
    return None


def detect_matmul_pattern(
    graphs: list[object],
    host_function: object,
) -> MatmulPatternInfo | None:
    """Detect matmul pattern from device IR graphs.

    Checks for nested tile loops (ForLoopGraphInfo, not reduction) containing
    mm/addmm operations, with no epilogue ops in the outer loop.

    Returns MatmulPatternInfo if the pattern matches, None otherwise.
    """
    from .._compiler.device_ir import ForLoopGraphInfo
    from .._compiler.device_ir import ReductionLoopGraphInfo
    from .._compiler.device_ir import RootGraphInfo

    # Must have non-reduction ForLoopGraphInfo (nested tile loops = matmul)
    for_loop_graphs = [
        g
        for g in graphs
        if isinstance(g, ForLoopGraphInfo) and not isinstance(g, ReductionLoopGraphInfo)
    ]
    if not for_loop_graphs:
        return None

    # Look for mm/addmm/bmm/baddbmm in a ForLoopGraphInfo graph
    _MM_OPS_2D = {"mm", "addmm"}
    _MM_OPS_3D = {"bmm", "baddbmm"}
    _MM_OPS = _MM_OPS_2D | _MM_OPS_3D
    mm_node = None
    mm_op: str | None = None
    for g in for_loop_graphs:
        for node in g.graph.nodes:
            if node.op == "call_function":
                op = getattr(node.target, "_opname", None)
                if op in _MM_OPS:
                    mm_node = node
                    mm_op = op
                    break
        if mm_node is not None:
            break

    if mm_node is None or mm_op is None:
        return None
    batched = mm_op in _MM_OPS_3D

    # Reject epilogue ops: check the RootGraphInfo (outer loop) for
    # computation between _phi (loop result) and store. A pure matmul
    # has only _phi -> store; an epilogue adds ops like relu, add, etc.
    _INFRA_OPS = {
        "_host_tensor",
        "_get_symnode",
        "_for_loop",
        "_phi",
        "_new_var",
        "full",
        "getitem",
        "load",
        "store",
        "sym_size.int",
    }
    for g in graphs:
        if not isinstance(g, RootGraphInfo):
            continue
        for node in g.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            name = getattr(target, "__name__", None) or getattr(target, "_opname", None)
            if name is not None and name not in _INFRA_OPS:
                return None  # Has epilogue / extra computation

    # Trace operands back through load -> _host_tensor to get tensor names.
    # addmm/baddbmm: (acc, lhs, rhs); mm/bmm: (lhs, rhs)
    if mm_op in ("addmm", "baddbmm"):
        a_load, b_load = mm_node.args[1], mm_node.args[2]
    else:
        a_load, b_load = mm_node.args[0], mm_node.args[1]

    def trace_load_to_name(load_node: object) -> str | None:
        if not isinstance(load_node, Node):
            return None
        if load_node.op != "call_function":
            return None
        if getattr(load_node.target, "__name__", "") != "load":
            return None
        host_node = load_node.args[0]
        if not isinstance(host_node, Node):
            return None
        return _get_host_tensor_name(host_node)

    a_name = trace_load_to_name(a_load)
    b_name = trace_load_to_name(b_load)
    if a_name is None or b_name is None:
        return None

    # Verify these are the only two tensor params (no hidden inputs)
    tensor_param_names = [
        name
        for name, val in host_function.params.arguments.items()
        if isinstance(val, torch.Tensor)
    ]
    if set(tensor_param_names) != {a_name, b_name}:
        return None

    return MatmulPatternInfo(
        a_name=a_name,
        b_name=b_name,
        input_names=tensor_param_names,
        batched=batched,
    )


def generate_matmul_backward_source(info: MatmulPatternInfo) -> str:
    """Generate a Helion backward kernel for matmul (2D) or bmm (3D).

    Produces two matmul phases:
      grad_a = grad_out @ b.T  (reduce over N)
      grad_b = a.T @ grad_out  (reduce over M)
    """
    a = info.a_name
    b = info.b_name
    params = ", ".join(f"{n}: torch.Tensor" for n in info.input_names)

    if info.batched:
        # 3D: baddbmm with batch dim, permute transposes last two dims
        return f'''\
"""
Auto-generated Helion backward kernel.
"""

import torch
import helion
from helion import language as hl

@helion.kernel()
def backward_kernel(grad_out: torch.Tensor, {params}) -> tuple[torch.Tensor, ...]:
    batch, m, k = {a}.shape
    _, _, n = {b}.shape
    grad_{a} = torch.empty([batch, m, k], dtype={a}.dtype, device={a}.device)
    grad_{b} = torch.empty([batch, k, n], dtype={b}.dtype, device={b}.device)
    for tile_b, tile_m, tile_k in hl.tile([batch, m, k]):
        acc = hl.zeros([tile_b, tile_m, tile_k], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = torch.baddbmm(acc, grad_out[tile_b, tile_m, tile_n], {b}[tile_b, tile_k, tile_n].permute(0, 2, 1))
        grad_{a}[tile_b, tile_m, tile_k] = acc
    for tile_b2, tile_k2, tile_n2 in hl.tile([batch, k, n]):
        acc2 = hl.zeros([tile_b2, tile_k2, tile_n2], dtype=torch.float32)
        for tile_m2 in hl.tile(m):
            acc2 = torch.baddbmm(acc2, {a}[tile_b2, tile_m2, tile_k2].permute(0, 2, 1), grad_out[tile_b2, tile_m2, tile_n2])
        grad_{b}[tile_b2, tile_k2, tile_n2] = acc2
    return grad_{a}, grad_{b}
'''

    # 2D: addmm, permute transposes the two dims
    return f'''\
"""
Auto-generated Helion backward kernel.
"""

import torch
import helion
from helion import language as hl

@helion.kernel()
def backward_kernel(grad_out: torch.Tensor, {params}) -> tuple[torch.Tensor, ...]:
    m, k = {a}.shape
    _, n = {b}.shape
    grad_{a} = torch.empty([m, k], dtype={a}.dtype, device={a}.device)
    grad_{b} = torch.empty([k, n], dtype={b}.dtype, device={b}.device)
    for tile_m, tile_k in hl.tile([m, k]):
        acc = hl.zeros([tile_m, tile_k], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = torch.addmm(acc, grad_out[tile_m, tile_n], {b}[tile_k, tile_n].permute(1, 0))
        grad_{a}[tile_m, tile_k] = acc
    for tile_k2, tile_n2 in hl.tile([k, n]):
        acc2 = hl.zeros([tile_k2, tile_n2], dtype=torch.float32)
        for tile_m2 in hl.tile(m):
            acc2 = torch.addmm(acc2, {a}[tile_m2, tile_k2].permute(1, 0), grad_out[tile_m2, tile_n2])
        grad_{b}[tile_k2, tile_n2] = acc2
    return grad_{a}, grad_{b}
'''


def backward(
    kernel: Kernel[object],
    grad_out: torch.Tensor,
    *inputs: torch.Tensor,
    return_code: bool = False,
    autotune: bool = False,
    autotune_effort: str | None = None,
) -> (
    tuple[torch.Tensor, ...]
    | torch.Tensor
    | tuple[tuple[torch.Tensor, ...] | torch.Tensor, str, str]
):
    """
    Compute gradients for a Helion kernel.

    The backward kernel is generated as an independent Helion kernel with its
    own ConfigSpec, allowing separate autotuning from the forward kernel.

    Args:
        kernel: A @helion.kernel decorated function (must be called once first)
        grad_out: Gradient of loss w.r.t. kernel output
        *inputs: The original inputs to the kernel (in the same order as forward)
        return_code: If True, also return the generated backward kernel code
        autotune: If True, autotune the backward kernel for best performance
        autotune_effort: Autotuning effort level ('none', 'quick', 'full').
            Default is 'none' when autotune=False, 'quick' when autotune=True.

    Returns:
        If return_code=False: Tuple of gradients (or single tensor if one input)
        If return_code=True: (gradients, helion_code, triton_code) tuple

    Example:
        @helion.kernel()
        def my_kernel(x, y):
            out = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                out[tile] = x[tile] * y[tile]
            return out

        out = my_kernel(x, y)
        grad_x, grad_y = helion.experimental.backward(my_kernel, grad_out, x, y)
    """
    if not hasattr(kernel, "_bound_kernels") or not kernel._bound_kernels:
        raise exc.AutodiffKernelNotCalled

    bound = kernel.bind(inputs)
    if bound._config is None:
        bound._config = bound.env.config_spec.default_config()

    if bound._backward_compiled is not None:
        bwd_fn, bwd_source, bwd_bound = bound._backward_compiled
    else:
        from .._compiler.device_ir import ForLoopGraphInfo
        from .._compiler.device_ir import ReductionLoopGraphInfo
        from .._compiler.device_ir import RootGraphInfo

        host_function = bound.host_function
        assert host_function is not None
        graphs = host_function.device_ir.graphs

        # Try matmul pattern first (nested tile loops with mm/addmm)
        matmul_info = detect_matmul_pattern(graphs, host_function)
        if matmul_info is not None:
            bwd_source = generate_matmul_backward_source(matmul_info)
        else:
            # Reject kernels with non-reduction ForLoopGraphInfo (multiple tile loops)
            for graph_info in graphs:
                if isinstance(graph_info, ForLoopGraphInfo) and not isinstance(
                    graph_info, ReductionLoopGraphInfo
                ):
                    raise exc.AutodiffNotSupported("multiple tile loops")

            # Find the RootGraphInfo — inline reductions produce additional
            # ReductionLoopGraphInfo graphs alongside the root, which we ignore.
            root_graphs = [g for g in graphs if isinstance(g, RootGraphInfo)]
            if len(root_graphs) != 1:
                raise exc.AutodiffNotSupported("multiple graphs")

            fwd_graph = root_graphs[0].graph

            analyzer = GraphAnalyzer(fwd_graph)
            compute_graph, input_mappings, output_mappings = (
                analyzer.extract_computation_graph()
            )

            # Reject keepdim=True reductions: grad_out would have the same ndim
            # as inputs, making the reduction invisible to our backward codegen.
            grad_out_ndim = len(grad_out.shape)
            max_input_ndim = max(len(t.shape) for t in inputs)
            if grad_out_ndim == max_input_ndim and any(
                node.target
                for node in fwd_graph.nodes
                if node.op == "call_function"
                and getattr(node.target, "_opname", None)
                in ("sum", "mean", "amax", "amin")
            ):
                # Check if any reduction ops exist — if so, this is likely
                # keepdim=True which we don't support (grad_out shape matches
                # input shape but backward needs reduction-aware iteration).
                for node in fwd_graph.nodes:
                    if node.op != "call_function":
                        continue
                    op = getattr(node.target, "_opname", None)
                    if op in ("sum", "mean", "amax", "amin"):
                        # Check if output shape differs from input shape
                        val = node.meta.get("val")
                        if val is not None and val.ndim == max_input_ndim:
                            raise exc.AutodiffNotSupported("keepdim=True reductions")

            backward_graph = differentiate_graph(compute_graph, inputs)

            converter = FXToHelionConverter(
                backward_graph=backward_graph,
                input_mappings=input_mappings,
                input_tensors=inputs,
                grad_out_shape=tuple(grad_out.shape),
            )
            bwd_source = converter.convert()

        with tempfile.TemporaryDirectory(prefix="helion_bwd_") as cache_dir:
            source_hash = hashlib.md5(
                bwd_source.encode(), usedforsecurity=False
            ).hexdigest()[:12]
            temp_path = pathlib.Path(cache_dir) / f"helion_bwd_{source_hash}.py"

            temp_path.write_text(bwd_source)

            spec = importlib.util.spec_from_file_location(
                f"helion_bwd_{source_hash}", str(temp_path)
            )
            assert spec is not None and spec.loader is not None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            assert hasattr(module, "backward_kernel")

            bwd_fn = module.backward_kernel
            bwd_args = (grad_out, *inputs)
            bwd_bound = bwd_fn.bind(bwd_args)

            # Determine autotune_effort: use 'quick' when autotuning, 'none' otherwise
            if autotune_effort is None:
                autotune_effort = "quick" if autotune else "none"

            # Set autotune_effort to prevent automatic autotuning on first call
            bwd_bound.settings.autotune_effort = autotune_effort
            if autotune:
                bwd_bound.autotune(bwd_args)
            bound._backward_compiled = (bwd_fn, bwd_source, bwd_bound)

    result = bwd_fn(grad_out, *inputs)
    if isinstance(result, tuple):
        assert all(isinstance(r, torch.Tensor) for r in result)
        grads: torch.Tensor | tuple[torch.Tensor, ...] = (
            result if len(result) > 1 else result[0]
        )
    else:
        assert isinstance(result, torch.Tensor)
        grads = result

    if return_code:
        if bwd_bound._config is None:
            bwd_bound._config = bwd_bound.env.config_spec.default_config()
        triton_code: str = bwd_bound.to_triton_code(bwd_bound._config)
        return grads, bwd_source, triton_code

    return grads
