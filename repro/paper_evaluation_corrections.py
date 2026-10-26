"""Frozen-evaluation-only interventions. Never changes weights or simulator source."""
import torch


def stable_action_scores(ctrl, net, state_batch):
    """Return E[Z - Vmin] in float64 for argmax, retaining float32 network logits.

    Scores omit the common Vmin constant solely to avoid cancellation. They are
    ranking scores, not the unshifted Q-values, and must not be used for training.
    """
    assert not ctrl.enable_learning and not net.training
    _, logits = net(state_batch, return_logits=True)
    probs = torch.softmax(logits.double(), dim=-1)
    centered = ctrl.support.double() - ctrl.support[0].double()
    return (probs * centered.view(1, 1, -1)).sum(-1)


def preserve_recovery_queue(command, state, mm, inv_limit):
    """Only suppress an exact same-price cancel/repost of the sole live order.

    Keep original policy/throttle call. Never modify a changed-price/missing-order
    command, never skip a forbidden-side cancellation, never invoke smart-place
    fallback on a newly requested price. The engine prunes stale orders before
    invoking policy; additionally require the state's live-order flag.
    """
    if mm is None or not isinstance(command, tuple) or not command:
        return command
    inv = float(state.get('inventory', 0.))
    if inv <= -inv_limit:
        side, good, forbidden, tag = 1, 'has_bid', 'has_ask', 'cancel_bid_then_place'
    elif inv >= inv_limit:
        side, good, forbidden, tag = -1, 'has_ask', 'has_bid', 'cancel_ask_then_place'
    else:
        return command
    if not state.get(good, False) or state.get(forbidden, False):
        return command
    if command[0] == 'cancel_all_then_place' and len(command) == 3:
        if int(command[1]) != side:
            return command
        requested = command[2]
    elif command[0] == tag and len(command) == 2:
        requested = command[1]
    else:
        return command
    if len(mm.orders) != 1:
        return command
    order = next(iter(mm.orders.values()))
    if int(order['side']) != side:
        return command
    if float(order.get('qty', 0.0)) != 1.0:
        return command
    target = int(requested)
    if not mm.backtest_mode:
        target = max(0, min(target, mm.owner.shape[1] - 1))
    if int(order['price']) != target:
        return command
    return ('hold',)


def instrument_controller(ctrl, variant, counters):
    assert variant == 'both', 'Only final-paper controller evaluation is supported.'
    if variant in ('stable_only', 'both'):
        ctrl._q_values_from_net = lambda net, states: stable_action_scores(ctrl, net, states)
    original_act = ctrl.act
    def act(state, mm=None):
        command = original_act(state)
        if ctrl._last_was_decision:
            counters['decisions'] += 1
        if variant in ('queue_only', 'both'):
            altered = preserve_recovery_queue(command, state, mm, float(ctrl.inv_limit))
            if altered != command:
                counters['preserved_recovery_quotes'] += 1
            return altered
        return command
    ctrl.act = act


def instrument_benchmark(policy, inv_limit, counters):
    def wrapped(state, mm=None):
        command = policy(state, mm=mm)
        altered = preserve_recovery_queue(command, state, mm, inv_limit)
        if altered != command:
            counters['preserved_recovery_quotes'] += 1
        return altered
    return wrapped
