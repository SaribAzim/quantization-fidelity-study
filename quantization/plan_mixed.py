"""
Mixed precision, step 2: turn the sensitivity measurements into build plans.

Upgrading a weight table from 4 to 8 bits costs half a byte per number. We give
both candidate layouts the SAME extra-size budget, so the comparison is fair:

  layers plan: the most sensitive whole layers get all 7 tables at 8-bit
  types plan:  the most sensitive KIND of table (per number, so big tables
               are not favoured) gets 8-bit in every layer, then the next kind,
               and so on; if a kind does not fit entirely, it is upgraded only
               in the most sensitive layers until the budget is used up

Run:  python quantization/plan_mixed.py [extra GB budget]
Writes results/mixed/plan_layers.json and results/mixed/plan_types.json
"""

import json
import sys

HIDDEN, INTERMEDIATE, KV = 4096, 14336, 1024
PARAMS = {  # numbers in one weight table of each kind
    "self_attn.q_proj": HIDDEN * HIDDEN, "self_attn.k_proj": KV * HIDDEN, "self_attn.v_proj": KV * HIDDEN,
    "self_attn.o_proj": HIDDEN * HIDDEN, "mlp.gate_proj": INTERMEDIATE * HIDDEN,
    "mlp.up_proj": INTERMEDIATE * HIDDEN, "mlp.down_proj": HIDDEN * INTERMEDIATE,
}
GB = 1024 ** 3


def extra_gb(params):
    return params * 0.5 / GB  # 8-bit instead of 4-bit: half a byte more per number


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    sens = json.load(open("results/mixed/sensitivity.json"))
    layer_rank = sorted(range(32), key=lambda l: -sens["per_layer"][str(l)]["kl_change"])
    n_test = len(sens["type_test_layers"])

    # --- layers plan
    per_layer_cost = extra_gb(sum(PARAMS.values()))
    chosen_layers, spent = [], 0.0
    for l in layer_rank:
        if spent + per_layer_cost <= budget:
            chosen_layers.append(l)
            spent += per_layer_cost
    plan_layers = {f"model.layers.{l}.{t}": 8 for l in chosen_layers for t in PARAMS}
    # A much cheaper third option: protect only the 3 most sensitive layers (the data shows 3 stand out).
    top3 = layer_rank[:3]
    plan_top3 = {f"model.layers.{l}.{t}": 8 for l in top3 for t in PARAMS}

    # --- types plan
    density = {t: sens["per_type"][t]["kl_change"] / (n_test * PARAMS[t]) for t in PARAMS}
    type_rank = sorted(PARAMS, key=lambda t: -density[t])
    plan_types, spent_t, summary = {}, 0.0, []
    for t in type_rank:
        for l in layer_rank:  # most sensitive layers first, so a partial fill still goes where it matters
            if spent_t + extra_gb(PARAMS[t]) <= budget:
                plan_types[f"model.layers.{l}.{t}"] = 8
                spent_t += extra_gb(PARAMS[t])
        summary.append(f"{t}: {sum(1 for k in plan_types if k.endswith(t))}/32 layers")

    for name, plan, cost, desc in [("layers", plan_layers, spent, f"layers {sorted(chosen_layers)}"),
                                   ("types", plan_types, spent_t, "; ".join(summary)),
                                   ("top3", plan_top3, 3 * per_layer_cost, f"layers {sorted(top3)}")]:
        json.dump({"overrides": plan, "extra_gb": cost, "description": desc}, open(f"results/mixed/plan_{name}.json", "w"), indent=1)
        print(f"plan '{name}': {len(plan)} tables at 8-bit, +{cost:.2f} GB  ({desc})")


if __name__ == "__main__":
    main()
