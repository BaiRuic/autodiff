
import graphviz

def label_info(node):
    if node.op is None:
        info = f"{str(node.realized_cached_data())}"
    else:
        info = f"{node.op.__class__.__name__}\n{str(node.realized_cached_data())}"
    return info
def draw_graph(node):
    dot = graphviz.Digraph()
    visited = set()

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        with dot.subgraph() as s:
            s.attr(rank='same') 
            dot.node(str(id(node)), label=label_info(node))
            if node.grad is not None:
                visited.add(node)
                with s.subgraph() as t:
                    t.attr(rank='same')
                    t.node(str(id(node.grad)), shape="square", label=label_info(node.grad))
                    t.edge(str(id(node)), str(id(node.grad)))
                    for input_grad_node in node.grad.inputs:
                        dot.edge(str(id(input_grad_node)), str(id(node.grad)))
        # 构建边
        for input_node in node.inputs:
            dot.edge(str(id(input_node)), str(id(node)))
            visit(input_node)

    visit(node)
    return dot
