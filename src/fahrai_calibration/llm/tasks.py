"""Fixed synthetic tasks with two independent exact solvers."""

import ast
import itertools
import operator
import random
from fractions import Fraction

from .io import canonical

SEED = 20260922
FAMILIES = ("arithmetic", "state_trace", "ordered_rules", "constrained_choice")
RULES = {
    "arithmetic": (
        "Evaluate parentheses first. Multiplication and division precede addition "
        "and subtraction. Division is exact rational division, not rounding or "
        "integer truncation."
    ),
    "state_trace": (
        "Perform iterations in order. Within each iteration, update x first; then "
        "use the NEW x to update the accumulator. Modulo returns the nonnegative "
        "remainder."
    ),
    "ordered_rules": (
        "Inspect rules in the given order and return the result of the FIRST "
        "matching rule. Within a rule, all listed conditions must hold (AND). If no "
        "rule matches, return the default result."
    ),
    "constrained_choice": (
        "Each item may be selected at most once. Respect BOTH the total weight limit"
        " and the total cost limit. Include at least one A item and at least one B "
        "item. Maximize the sum of values and return that maximum value, not a list "
        "of item IDs."
    ),
}
INSTRUCTION = (
    "Solve the supplied exercise and briefly explain the method in English (at "
    "most 100 words). No learner answer has been observed. Return final_answer "
    "as an integer, explanation as a nonempty string, and source_ids citing the "
    "supplied rule ID. Do not claim learning gains. For an optimization "
    "exercise, final_answer is the maximum achievable total value. Return only "
    "the specified JSON object."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "final_answer": {"type": "integer"},
        "explanation": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["final_answer", "explanation", "source_ids"],
}


def arithmetic(r, level):
    terms = []
    for _ in range(level + 1):
        a, b, c = (r.randint(12, 95), r.randint(2, 13), r.randint(2, 20))
        terms.append((f"(({a} * {b}) + {b * c}) / {b}", a + c))
    signs = [r.choice([1, -1]) for _ in terms[1:]]
    expression = "(" + terms[0][0] + ")"
    for sign, (term, _) in zip(signs, terms[1:]):
        expression += (" + " if sign == 1 else " - ") + "(" + term + ")"
    if level == 3:
        expression = "(" + expression + ") * " + str(r.randint(2, 7))
    return {"expression": expression}


def state_trace(r, level):
    return {
        "x_initial": r.randint(1, 40),
        "acc_initial": r.randint(-10, 10),
        "a": r.randint(2, 7),
        "b": r.randint(1, 11),
        "modulus": r.choice([31, 37, 41, 43]),
        "iterations": {1: 3, 2: 6, 3: 10}[level],
    }


def ordered_rules(r, level):
    n = {1: 4, 2: 8, 3: 12}[level]
    features = {f"x{i}": r.randint(1, 60) for i in range(1, level + 3)}
    winning = r.randrange(n + 1)
    rules = []
    for idx in range(n):
        atoms = []
        for j in range(level + 1):
            name = r.choice(list(features))
            value = features[name]
            op = r.choice(["<=", ">=", "=="])
            truth = idx == winning or j > 0 or idx > winning
            if idx > winning:
                truth = bool(r.randrange(2))
            threshold = {
                "<=": value + r.randint(0, 8) if truth else value - r.randint(1, 8),
                ">=": value - r.randint(0, 8) if truth else value + r.randint(1, 8),
                "==": value if truth else value + r.randint(1, 8),
            }[op]
            atoms.append({"feature": name, "operator": op, "threshold": threshold})
        rules.append({"conditions_all": atoms, "result": idx + 1})
    return {"features": features, "rules": rules, "default_result": 0}


def constrained_choice(r, level):
    n = {1: 4, 2: 7, 3: 10}[level]
    items = [
        {
            "id": i + 1,
            "type": "A" if i % 2 == 0 else "B",
            "weight": r.randint(1, 9),
            "cost": r.randint(1, 12),
            "value": r.randint(3, 30),
        }
        for i in range(n)
    ]
    return {
        "items": items,
        "weight_limit": max(
            items[0]["weight"] + items[1]["weight"], int(sum((i["weight"] for i in items)) * 0.48)
        ),
        "cost_limit": max(
            items[0]["cost"] + items[1]["cost"], int(sum((i["cost"] for i in items)) * 0.48)
        ),
    }


GEN = dict(zip(FAMILIES, (arithmetic, state_trace, ordered_rules, constrained_choice)))
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def eval_ast(n):
    if isinstance(n, ast.Expression):
        return eval_ast(n.body)
    if isinstance(n, ast.Constant) and type(n.value) is int:
        return Fraction(n.value)
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        return OPS[type(n.op)](eval_ast(n.left), eval_ast(n.right))
    raise ValueError("unsupported arithmetic")


def postfix_eval(expression):
    import re

    tokens = re.findall("\\d+|[()+*/-]", expression)
    stack = []
    output = []
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    for t in tokens:
        if t.isdigit():
            output.append(t)
        elif t == "(":
            stack.append(t)
        elif t == ")":
            while stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != "(" and (precedence[stack[-1]] >= precedence[t]):
                output.append(stack.pop())
            stack.append(t)
    output.extend(reversed(stack))
    values = []
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
    for t in output:
        if t.isdigit():
            values.append(Fraction(int(t)))
        else:
            b = values.pop()
            a = values.pop()
            values.append(ops[t](a, b))
    assert len(values) == 1
    return values[0]


def solve_primary(f, d):
    if f == "arithmetic":
        return eval_ast(ast.parse(d["expression"], mode="eval"))
    if f == "state_trace":
        x, acc = (d["x_initial"], d["acc_initial"])
        for _ in range(d["iterations"]):
            x = (d["a"] * x + d["b"]) % d["modulus"]
            acc += x if x % 2 == 0 else -x
        return acc
    if f == "ordered_rules":
        ops = {"<=": operator.le, ">=": operator.ge, "==": operator.eq}
        for rule in d["rules"]:
            if all(
                (
                    ops[c["operator"]](d["features"][c["feature"]], c["threshold"])
                    for c in rule["conditions_all"]
                )
            ):
                return rule["result"]
        return d["default_result"]
    if f == "constrained_choice":
        best = None
        items = d["items"]
        for bits in itertools.product((False, True), repeat=len(items)):
            selected = [i for i, b in zip(items, bits) if b]
            if {i["type"] for i in selected} != {"A", "B"}:
                continue
            if (
                sum((i["weight"] for i in selected)) > d["weight_limit"]
                or sum((i["cost"] for i in selected)) > d["cost_limit"]
            ):
                continue
            value = sum((i["value"] for i in selected))
            best = value if best is None else max(best, value)
        if best is None:
            raise ValueError("infeasible")
        return best
    raise ValueError(f)


def solve_secondary(f, d):
    if f == "arithmetic":
        return postfix_eval(d["expression"])
    if f == "state_trace":

        def rec(k, x):
            if k == 0:
                return 0
            nxt = d["a"] * x + d["b"]
            nxt = nxt - nxt // d["modulus"] * d["modulus"]
            return (nxt if nxt // 2 * 2 == nxt else -nxt) + rec(k - 1, nxt)

        return d["acc_initial"] + rec(d["iterations"], d["x_initial"])
    if f == "ordered_rules":
        matches = []
        for idx, rule in enumerate(d["rules"]):
            failed = False
            for c in rule["conditions_all"]:
                delta = d["features"][c["feature"]] - c["threshold"]
                if (
                    c["operator"] == "<="
                    and delta > 0
                    or (c["operator"] == ">=" and delta < 0)
                    or (c["operator"] == "==" and delta != 0)
                ):
                    failed = True
            if not failed:
                matches.append((idx, rule["result"]))
        return min(matches)[1] if matches else d["default_result"]
    if f == "constrained_choice":
        states = {(0, 0, 0): 0}
        for i in d["items"]:
            nxt = dict(states)
            for (w, c, mask), v in states.items():
                nw, nc = (w + i["weight"], c + i["cost"])
                key = (nw, nc, mask | (1 if i["type"] == "A" else 2))
                if nw <= d["weight_limit"] and nc <= d["cost_limit"]:
                    nxt[key] = max(nxt.get(key, -1), v + i["value"])
            states = nxt
        return max((v for (w, c, mask), v in states.items() if mask == 3))
    raise ValueError(f)


def render(f, d):
    if f == "arithmetic":
        return "Compute exactly: " + d["expression"] + "."
    if f == "state_trace":
        return (
            f"Set x = {d['x_initial']} and accumulator = {d['acc_initial']}. "
            f"Repeat exactly {d['iterations']} times: (1) x = ({d['a']} * x + "
            f"{d['b']}) modulo {d['modulus']}; (2) if the new x is even, "
            "add x to accumulator; otherwise subtract x from accumulator. "
            "What is the final accumulator?"
        )
    if f == "ordered_rules":
        return "Apply the first matching rule. Data: " + canonical(d)
    if f == "constrained_choice":
        return (
            "Choose a subset of the items and return its maximum total value under the "
            "stated rules and limits. Data: "
        ) + canonical(d)


def schedule(tasks):
    rng = random.Random(20260923)
    jobs = []
    for phase, repeats in [("development", 1), ("main", 2)]:
        for rep in range(1, repeats + 1):
            ids = [t["task_id"] for t in tasks if t["phase"] == phase]
            rng.shuffle(ids)
            for task_id in ids:
                jobs.append(
                    {
                        "assignment_id": task_id + f":r{rep}",
                        "phase": phase,
                        "task_id": task_id,
                        "repeat": rep,
                    }
                )
    return jobs


def generate(population):
    rng = random.Random(SEED)
    cells = {
        int(row["cell_id"]): {
            "cell_id": int(row["cell_id"]),
            "state": float(row["state"]),
            "regime": int(row["regime"]),
            "action": int(row["action"]),
            "target_mass": float(row["target_mass"]),
            "true_learner_probability": float(row["probability"]),
        }
        for row in population
    }
    tasks = []
    for phase, ids in [("development", [None]), ("main", list(range(12)))]:
        for cell_id in ids:
            for family in FAMILIES:
                for level in (1, 2, 3):
                    data = GEN[family](rng, level)
                    answer = solve_primary(family, data)
                    if answer != solve_secondary(family, data) or int(answer) != answer:
                        raise ValueError("Reference solvers disagree")
                    cell_name = "global" if cell_id is None else f"c{cell_id:02d}"
                    task = {
                        "task_id": f"{phase[:3]}-{cell_name}-{family}-L{level}",
                        "phase": phase,
                        "family": family,
                        "structural_level": level,
                        "data": data,
                        "exercise": render(family, data),
                        "gold_answer": int(answer),
                        "reference": {"source_id": "RULE." + family.upper(), "text": RULES[family]},
                        "label_provenance": (
                            "two independent deterministic solvers; no expert or LLM grading"
                        ),
                    }
                    if cell_id is not None:
                        task.update(cells[cell_id])
                    tasks.append(task)
    return tasks
