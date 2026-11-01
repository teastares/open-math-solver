"""独立只读复核v98三账本与一致搜索尺度，不运行求解或新转换器。"""
import collections
import hashlib
import json
from pathlib import Path
import types

# 本轮只读源码证明在文件末尾；账本仍按父schema3核验。

PLAN_SHA256 = "a66d1fc30ba2cc8eb2a9a56fe1381801b6f73297fabc9e2cb8bd6ae1e2f219a2"
CLARIFICATION_SHA256 = "be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8"
V92_AUDIT_SHA256 = "948a7ea3078ef5f1a20251624d92832ec73be1d24fb4209c175969b9fb472fb2"
V92_INST_SHA256 = "9521e91350bf1082cf8ada80a922f3c36e0ef419f91170a2bf1227ad80733762"
COUPLED_KERNEL_SHA256 = "601878a7525336108d126e57ad6fd326e1ab7685eb2dc61421e73a6b666c2017"
ENERGY_COUNTS = ("started", "completed", "primal_terms", "row_terms", "budget_checks")
ENERGY_FIELDS = {"schema", "enabled", *ENERGY_COUNTS, "last"}
ENERGY_LAST_FIELDS = {"attempt_iteration", "status", "columns", "rows", "primal_regularization", "primal_terms",
    "row_terms", "primal_energy", "row_energy", "denominator"}
ENERGY_STATUSES = {"running", "completed", "numerical_error", "exception", "budget"}
REFERENCE_VARIANTS = {"frozen_v66", "highs"}
_PARENT = None


def require(value, message):
    if not value:
        raise ValueError(message)


def parent_auditor():
    """只载入已封存旧审计核；原21与FG16检查不依赖本轮runner。"""
    global _PARENT
    if _PARENT is None:
        path = Path(__file__).with_name("v92-coupled-audit.py")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == V92_AUDIT_SHA256, "冻结v92纯读审计来源变化")
        module = types.ModuleType("v98_frozen_coupled_audit")
        module.__file__ = str(path)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        _PARENT = module
    return _PARENT


def empty_energy(enabled=False):
    return dict(schema=1, enabled=enabled, **dict.fromkeys(ENERGY_COUNTS, 0), last=None)


def independent_energy(value, enabled):
    old = parent_auditor()
    require(isinstance(value, dict) and set(value) == ENERGY_FIELDS and type(value["schema"]) is int and value["schema"] == 1,
        "能量账本精确8字段及schema")
    require(type(value["enabled"]) is bool and value["enabled"] is enabled and all(old.integer(value[key]) for key in ENERGY_COUNTS),
        "能量作用域布尔与非负整数计数")
    if not enabled:
        require(old.equal(value, empty_energy()), "关闭或旁路能量策略必须零工作与null末快照")
        return
    started, completed = value["started"], value["completed"]
    require(0 <= started - completed <= 1 and started <= 200, "每外层一次能量归约，至多末调用未交付")
    last = value["last"]
    require((last is None) == (started == 0), "能量末快照存在当且仅当实际启动过helper")
    if last is None:
        require(all(value[key] == 0 for key in ENERGY_COUNTS), "未进helper不得冒记项或预算检查")
        return
    require(isinstance(last, dict) and set(last) == ENERGY_LAST_FIELDS and old.integer(last["attempt_iteration"], 200) and
        last["attempt_iteration"] > 0 and type(last["status"]) is str and last["status"] in ENERGY_STATUSES and
        all(old.integer(last[key]) for key in ("columns", "rows", "primal_terms", "row_terms")),
        "能量末快照精确10字段、实际外层标签与合法项计数")
    n, m, pt, rt = (last[key] for key in ("columns", "rows", "primal_terms", "row_terms"))
    require(pt <= n and rt <= m and (rt == 0 or pt == n), "全列阶段完成后才累计全部存储行")
    require((last["primal_regularization"] is None or old.finite(last["primal_regularization"])) and
        all(old.nonnegative(last[key]) for key in ("primal_energy", "row_energy", "denominator")),
        "LD能量摘要必须有限非负或窄化溢出的null，r失败摘要可保原输入")
    require((pt != 0 or last["primal_energy"] == 0) and (rt != 0 or last["row_energy"] == 0),
        "未付出的项不能形成能量")
    complete = last["status"] == "completed"
    require(complete == (started == completed), "只有helper正常末预算返回才增加completed")
    if complete:
        require(pt == n and rt == m and last["primal_regularization"] is not None and last["primal_regularization"] >= 0,
            "完整归约必须处理全量方向及每行实际R，r有限非负")
    denominator = last["denominator"]
    if denominator is not None:
        require(denominator >= 1 and pt == n and rt == m and last["primal_energy"] is not None and last["row_energy"] is not None and
            denominator >= max(last["primal_energy"], last["row_energy"]),
            "D只能在全项后形成；有有限D时两项能量也可表示且不大于D")
    require(value["primal_terms"] == completed * n + (0 if complete else pt) and
        value["row_terms"] == completed * m + (0 if complete else rt),
        "固定Cone内各完整调用和末partial项精确守恒")
    checks = 2 + (n + 4095) // 4096 + (m + 4095) // 4096
    base = completed * checks
    require(base <= value["budget_checks"] <= base + (0 if complete else checks),
        "入口/每4096批/末检查与实际调用前缀相容")


def independent_energy_public(energy, coupled, report):
    old = parent_auditor()
    require(energy["enabled"] is coupled["enabled"], "能量与联立作用域必须同启停")
    if not energy["enabled"]:
        return
    require(0 <= coupled["constant_calls_completed"] - energy["started"] <= 1 and
        0 <= coupled["constant_calls_started"] - energy["completed"] <= 1 and
        coupled["directions_started"] <= 2 * energy["completed"],
        "真实常量direct后才归约能量，能量正常返回后才预测/校正；仅末预算入口可差一")
    if energy["last"] is not None:
        require(old.integer(report.get("ipm_iterations"), 200) and energy["last"]["attempt_iteration"] <= report["ipm_iterations"] + 1,
            "能量本次尝试编号至多比已提交HSD点大1")
        if coupled["last"] is not None:
            last, direction = energy["last"], coupled["last"]
            require(last["columns"] + last["rows"] + 2 <= direction["dimension"] <= last["columns"] + 2 * last["rows"] + 2,
                "真实全五块方向维数与固定Cone列行数相容")


def _inspect(row):
    old = parent_auditor()
    oracle = old.parent_auditor()
    variant = row["requested_variant"]
    prefixes = ("V97_PRESOLVE ", "V97_COUPLED ", "V97_ENERGY ")
    lines = [line for line in row["stderr"].splitlines() if line.strip()]
    values, errors = [], []
    for prefix in prefixes:
        selected = [line for line in lines if line.startswith(prefix)]
        if variant in REFERENCE_VARIANTS:
            values.append(None); errors.append(None if not selected else "unexpected_marker")
        elif len(selected) != 1:
            values.append(None); errors.append("marker_count")
        else:
            try:
                values.append(oracle.parse(selected[0][len(prefix):])); errors.append(None)
            except (ValueError, TypeError):
                values.append(None); errors.append("invalid_marker_json")
    primary, coupled, energy = values
    if variant not in REFERENCE_VARIANTS:
        if errors[0] is None:
            try:
                oracle.independent_marker(primary, variant == "diagnostic_on")
            except (ValueError, TypeError, KeyError):
                errors[0] = "invalid_marker_schema"
        scope = variant == "diagnostic_on" and isinstance(primary, dict) and primary.get("status") not in {"disabled", "bypassed"}
        for index, checker in ((1, independent_coupled), (2, independent_energy)):
            if errors[index] is None:
                try:
                    checker(values[index], scope)
                except (ValueError, TypeError, KeyError):
                    errors[index] = "invalid_coupled_schema" if index == 1 else "invalid_energy_schema"
    require(all(any(line.startswith(prefix) for prefix in prefixes) for line in lines), "三份私有账本之外不得有未解释stderr")
    return primary, errors[0], coupled, errors[1], energy, errors[2]


def audit_rows(rows):
    """由原始stdout/stderr独立重算三marker、原oracle计时与真实线性工作前缀。"""
    old = parent_auditor()
    oracle = old.parent_auditor()
    for row in rows:
        marker, error, coupled, coupled_error, energy, energy_error = _inspect(row)
        require(old.equal(marker, row["presolve_marker"]) and error == row["presolve_marker_error"] and
            old.equal(coupled, row["coupled_marker"]) and coupled_error == row["coupled_marker_error"] and
            old.equal(energy, row["energy_marker"]) and energy_error == row["energy_marker_error"],
            "原始三marker、独立schema与冻结runner字段精确一致")
        report = row.get("report")
        if isinstance(report, dict):
            require(old.equal(oracle.parse(row["stdout"]), report), "公共报告必须对应真实stdout")
        if error is None and marker is not None:
            adapted = dict(row, presolve_marker=marker, presolve_marker_error=None,
                stderr="V88_PRESOLVE " + json.dumps(marker, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            oracle.audit_marker_rows([adapted])
        if coupled_error is not None or coupled is None or not isinstance(report, dict):
            continue
        independent_public(coupled, report, marker, energy)
        if energy_error is None and energy is not None:
            independent_energy_public(energy, coupled, report)
        if coupled["directions_started"]:
            require(marker is not None and marker["child"]["started"] == 1 and marker["route"] in {"reduced", "original"},
                "只有真实非空core能启动Newton核")
        if marker is not None and (marker["route"] == "empty" or marker["child"]["started"] == 0):
            require(all(coupled[key] == 0 for key in COUNTS) and
                (energy is None or all(energy[key] == 0 for key in ENERGY_COUNTS)), "空约化或未进core不得冒记线性或能量工作")


def diagnostic_summary(rows):
    rows = list(rows)
    audit_rows(rows)
    old = parent_auditor()
    on = [row for row in rows if row["requested_variant"] == "diagnostic_on"]
    coupled = [row["coupled_marker"] for row in on if row["coupled_marker_error"] is None and isinstance(row["coupled_marker"], dict)]
    energy = [row["energy_marker"] for row in on if row["energy_marker_error"] is None and isinstance(row["energy_marker"], dict)]
    primary = [row["presolve_marker"] for row in on if row["presolve_marker_error"] is None and isinstance(row["presolve_marker"], dict)]
    last = [value["last"] for value in coupled if value["last"] is not None]
    energy_last = [value["last"] for value in energy if value["last"] is not None]
    return dict(runs=len(rows), variant_counts=dict(collections.Counter(row["requested_variant"] for row in rows)),
        valid_on_marker_runs=len(coupled), active_scope_runs=sum(value["enabled"] for value in coupled),
        strict_kernel_return_sums=sum(value["directions_completed"] - value["initial_forcing_returns"] - value["candidate_forcing_returns"] for value in coupled),
        counter_sums={key: sum(value[key] for value in coupled) for key in COUNTS},
        last_status_counts=dict(collections.Counter(value["status"] for value in last)),
        last_role_counts=dict(collections.Counter(value["role"] for value in last)),
        energy_valid_on_marker_runs=len(energy), energy_active_scope_runs=sum(value["enabled"] for value in energy),
        energy_counter_sums={key: sum(value[key] for value in energy) for key in ENERGY_COUNTS},
        energy_last_status_counts=dict(collections.Counter(value["status"] for value in energy_last)),
        energy_last_snapshots=[dict(input=row["input"], repetition=row["repetition"], run_index=row["run_index"],
            last=row["energy_marker"]["last"]) for row in on if row["energy_marker_error"] is None and
            isinstance(row["energy_marker"], dict) and row["energy_marker"]["last"] is not None],
        parent_accepted_runs=sum(value["accepted"] for value in primary),
        oracle_events=sum(value["oracle"]["started"] for value in primary),
        original_quality_failure_facts=[{key: row.get(key) for key in ("input", "repetition", "run_index", "requested_variant",
            "status", "quality_pass", "quality_failure", "numerical_seconds", "elapsed_wall_seconds")} for row in rows if row.get("quality_pass") is not True],
        reused_readonly_auditor=dict(file=".tools/v92-coupled-audit.py", sha256=V92_AUDIT_SHA256),
        interpretation="accepted仅严格LD方向，forced_initial/forced_candidate是各自已验点的受控返回；三者都不是Point或LP成功。能量completed仅全量归约。"
            "Ddelta仍是近似P，未回算旧D且不能由本摘要声称改进量；全部能量工作计原epoch。"
            "末budget可保已形成D而completed0，有限LD摘要可窄0或溢double为null。"
            "原质量、全部失败、原owner及四ULP时钟由封存旧oracle独立复核，性能门槛由主结项重算。")


COUNTS = ('constant_calls_started', 'constant_calls_completed', 'directions_started', 'directions_completed',
    'preconditioner_started', 'preconditioner_completed', 'products_started', 'products_completed',
    'krylov_started', 'krylov_completed', 'cycles_started', 'true_checks_started', 'true_checks_completed',
    'initial_forcing_checks', 'initial_forcing_returns', 'candidate_forcing_checks', 'candidate_forcing_returns')
FIELDS = {'schema', 'enabled', *COUNTS, 'last'}
LAST_FIELDS = {'attempt_iteration', 'role', 'status', 'dimension', 'initial_measure', 'final_measure',
    'block_measures', 'tolerances', 'initial_forcing', 'candidate_forcing'}
FORCING_FIELDS = {'rhs_norms', 'defect_norms', 'measure', 'decision'}
STATUSES = {'running', 'accepted', 'forced_initial', 'forced_candidate', 'exhausted', 'breakdown', 'numerical_error', 'exception', 'budget'}
DECISIONS = {'not_checked', 'strict_initial', 'rejected', 'eligible', 'returned'}


def independent_coupled(value, enabled):
    old = parent_auditor()
    require(type(value) is dict and set(value) == FIELDS and type(value['schema']) is int and value['schema'] == 3,
        'schema3精确20键，不把旧schema1当forcing政策')
    require(type(value['enabled']) is bool and value['enabled'] is enabled and all(old.integer(value[key]) for key in COUNTS),
        '作用域布尔与17项非负整数计数')
    if not enabled:
        require(old.equal(value, dict(schema=3, enabled=False, **dict.fromkeys(COUNTS, 0), last=None)), '关闭/旁路零工作')
        return
    for stem in ('constant_calls', 'directions', 'preconditioner', 'products', 'krylov', 'true_checks'):
        require(0 <= value[stem + '_started'] - value[stem + '_completed'] <= 1, '实际回调至多末一份partial：' + stem)
    constants, directions, columns = (value[key] for key in ('constant_calls_started', 'directions_started', 'krylov_started'))
    checks, returned = value['initial_forcing_checks'], value['initial_forcing_returns']
    require(constants <= 200 and directions <= 400 and directions <= 2 * value['constant_calls_completed'], '原200步与两方向上界')
    require(0 <= returned <= checks <= directions and returned <= value['directions_completed'] and checks <= value['true_checks_completed'],
        '只有完整初始真验后才gate，只有kernel正常返回才force-return')
    candidate_checks, candidate_returns = value['candidate_forcing_checks'], value['candidate_forcing_returns']
    require(0 <= candidate_returns <= candidate_checks <= value['krylov_completed'] and
        returned + candidate_returns <= value['directions_completed'] and returned + candidate_returns <= checks and
        candidate_returns <= value['cycles_started'] and checks + candidate_checks <= value['true_checks_completed'],
        '候选gate须有完整列/真验，且先前初gate已拒绝')
    remaining = directions - returned
    require(columns <= 96 * remaining and value['cycles_started'] <= 3 * remaining and
        columns <= 32 * value['cycles_started'] and value['cycles_started'] <= columns + 1,
        'forced初P无FG；其余仍原32列3周期与单partial')
    require(value['preconditioner_started'] <= 97 * remaining + returned and
        value['products_started'] <= 193 * remaining + returned and value['true_checks_started'] <= 97 * remaining + returned,
        'forced各1P/1A/1true，其他方向原有限上界')
    require(0 <= directions + columns - value['preconditioner_started'] <= 1 and
        0 <= columns + value['true_checks_started'] - value['products_started'] <= 1,
        '每方向1初P，每列1P/1A，每true1A，不重复初解')
    require(value['true_checks_started'] <= directions + columns and value['directions_completed'] <= value['true_checks_completed'],
        '所有返回均有真实完整初始/候选检查')
    last = value['last']
    require((last is None) == (directions == 0), 'last存在当且仅当启动过方向')
    if last is None:
        require(all(value[key] == 0 for key in COUNTS if not key.startswith('constant_calls')), '无方向仅可有常量partial')
        return
    require(type(last) is dict and set(last) == LAST_FIELDS and old.integer(last['attempt_iteration'], 200) and
        last['attempt_iteration'] > 0 and last['role'] in {'affine', 'corrector'} and last['status'] in STATUSES and
        old.integer(last['dimension']) and last['dimension'] >= 2, 'schema3 last精确10键及实际调用站')
    require(all(old.nonnegative(last[key]) for key in ('initial_measure', 'final_measure')) and
        type(last['block_measures']) is list and len(last['block_measures']) == 5 and all(old.nonnegative(x) for x in last['block_measures']) and
        type(last['tolerances']) is list and len(last['tolerances']) == 5 and all(old.finite(x) and x > 0 for x in last['tolerances']),
        '严格量摘要仍五块有限非负或LD溢出null，tau有限正')
    if last['status'] != 'forced_candidate' and last['initial_measure'] is not None and last['final_measure'] is not None:
        require(last['final_measure'] <= last['initial_measure'], '最终best完整真验不退于初始')
    if all(x is not None for x in last['block_measures']):
        require(last['final_measure'] == max(last['block_measures']), 'final属于同一完整五块快照')
    if last['status'] in {'accepted', 'forced_initial', 'forced_candidate'}:
        require(value['directions_completed'] == directions, '两类正常kernel返回都计completed')
    else:
        require(value['directions_completed'] + 1 == directions, '失败/partial不伪完成')
    if last['status'] == 'accepted':
        require(last['final_measure'] is not None and last['final_measure'] <= 1 and
            all(x is not None and x <= 1 for x in last['block_measures']), 'accepted仍只五块strict')
    independent_candidate_snapshot(value, last)
    proof = last['initial_forcing']
    if proof is None:
        require(last['status'] not in {'accepted', 'forced_initial', 'forced_candidate'} and last['initial_measure'] is None,
            '活动production正常初始真验及返回必须有实际B3对象')
        return
    require(type(proof) is dict and set(proof) == FORCING_FIELDS and type(proof['rhs_norms']) is list and len(proof['rhs_norms']) == 3 and
        all(old.finite(x) and x >= 0 for x in proof['rhs_norms']) and proof['decision'] in DECISIONS and old.nonnegative(proof['measure']),
        'forcing精确4键、实际完整rhs的B3及合法decision')
    defects, decision = proof['defect_norms'], proof['decision']
    require(defects is None or type(defects) is list and len(defects) == 5 and all(old.nonnegative(x) for x in defects),
        'E5只能完整发布，摘要超double可null，不发布半数组')
    if decision == 'not_checked':
        require(proof['measure'] is None and last['status'] not in {'accepted', 'forced_initial', 'forced_candidate'}, '未作gate不能伪造接纳')
    elif decision == 'strict_initial':
        require(defects is not None and all(old.finite(x) for x in defects) and proof['measure'] is None and
            last['initial_measure'] is not None and last['initial_measure'] <= 1 and last['status'] in {'accepted', 'budget', 'exception'},
            'strict初P优先且不混算forcing gate')
    else:
        require(defects is not None and checks > 0 and (last['initial_measure'] is None or last['initial_measure'] >= 1),
            'gate只在初P严格失败后，失败LD可摘要为1或null')
        if decision == 'rejected':
            require((proof['measure'] is None or proof['measure'] >= 1) and last['status'] != 'forced_initial',
                '拒绝gate可进strictFG，但不能forced返回')
        else:
            require(all(old.finite(x) for x in defects) and proof['measure'] is not None and proof['measure'] <= 1,
                'eligible/returned实际LD gate过线，E受有限F约束')
            if decision == 'eligible':
                require(last['status'] in {'budget', 'exception'}, 'eligible只解释末预算未返回')
            else:
                require(last['status'] == 'forced_initial' and returned > 0 and
                    old.equal(last['initial_measure'], last['final_measure']) and
                    all(x is not None and x <= 1 for x in last['block_measures'][3:]),
                    'forced保初P且两个互补块仍strict')
    if last['status'] == 'forced_initial':
        require(decision == 'returned', 'forced必须由真实返回decision与计数支持')


def independent_candidate_snapshot(value, last):
    old = parent_auditor()
    proof = last['candidate_forcing']
    if proof is None:
        require(last['status'] != 'forced_candidate', '候选强制返回必须有其真实完整proof')
        return
    fields = {'cycle', 'krylov_step', 'defect_norms', 'strict_measure', 'best_measure_before', 'measure', 'decision'}
    require(type(proof) is dict and set(proof) == fields and old.integer(proof['cycle'], 3) and
        old.integer(proof['krylov_step'], 96) and proof['decision'] in {'not_checked', 'rejected', 'eligible', 'returned'} and
        all(old.nonnegative(proof[key]) for key in ('strict_measure', 'best_measure_before', 'measure')),
        '候选proof精确7键及整数位置、LD非负摘要')
    defects, decision = proof['defect_norms'], proof['decision']
    require(defects is None or type(defects) is list and len(defects) == 5 and all(old.nonnegative(x) for x in defects),
        '候选E5只能完整发布，partial保前次完整proof')
    if decision == 'not_checked':
        require(proof['cycle'] == proof['krylov_step'] == 0 and defects is None and
            all(proof[key] is None for key in ('strict_measure', 'best_measure_before', 'measure')) and last['status'] != 'forced_candidate',
            '未做候选gate对象必须空，不能冒记返回')
        return
    require(1 <= proof['cycle'] <= proof['krylov_step'] <= min(32, last['dimension']) * proof['cycle'] and
        proof['krylov_step'] <= value['krylov_completed'] and value['candidate_forcing_checks'] > 0 and defects is not None,
        '完整候选gate的位置必须属于已经完成的真实列')
    require((proof['strict_measure'] is None or proof['strict_measure'] >= 1) and
        (proof['best_measure_before'] is None or proof['best_measure_before'] >= 1),
        '只在strict失败候选及strict未过best上检查，LD>1可摘要1/null')
    if proof['best_measure_before'] is not None and last['initial_measure'] is not None:
        require(proof['best_measure_before'] <= last['initial_measure'], 'gate前最佳strict完整快照不退于初P')
    initial = last['initial_forcing']
    require(type(initial) is dict and initial.get('decision') == 'rejected', '候选能力只在初P gate已拒后进入FG')
    if decision == 'rejected':
        require((proof['measure'] is None or proof['measure'] >= 1) and last['status'] != 'forced_candidate',
            '拒绝候选可继续strict，但不能标强制返回')
        return
    require(proof['measure'] is not None and proof['measure'] <= 1 and all(old.finite(x) for x in defects),
        'eligible/returned必须达到同一有限F界；不以JSON舍入值作充分数学证书')
    if decision == 'eligible':
        require(last['status'] in {'budget', 'exception'}, 'eligible仅说明末预算前完整gate，不能冒作正常交付')
        return
    require(last['status'] == 'forced_candidate' and value['candidate_forcing_returns'] > 0 and
        old.equal(last['final_measure'], proof['strict_measure']) and
        all(x is not None and x <= 1 for x in last['block_measures'][3:]),
        'forced_candidate最终摘要必须来自实际返回候选，即使strict比原best差')


def independent_public(coupled, report, primary, energy):
    old = parent_auditor()
    if not coupled['enabled']:
        return
    for key in ('ipm_weighted_krylov_steps', 'ipm_homogeneous_constant_repairs', 'ipm_homogeneous_constant_repair_successes'):
        require(type(report.get(key)) is int and report[key] == 0, '不伪造旧K0/constant repair工作：' + key)
    for private, public in (('constant_calls', 'ipm_homogeneous_constant_solves'),
            ('preconditioner', 'ipm_homogeneous_direction_solves'), ('products', 'ipm_homogeneous_scaled_products')):
        require(old.integer(report.get(public)) and coupled[private + '_completed'] <= report[public] <= coupled[private + '_started'] and
            coupled[private + '_started'] - report[public] <= 1, '真实public/caller单partial前缀：' + private)
    refined = report.get('ipm_homogeneous_scaled_refinement_steps')
    require(old.integer(refined) and coupled['krylov_completed'] <= refined <= coupled['krylov_started'] and
        old.integer(report.get('ipm_refinement_steps')) and report['ipm_refinement_steps'] >= refined, '每列P如实refinement计数')
    initial = report['ipm_homogeneous_direction_solves'] - refined
    require(coupled['directions_completed'] <= initial <= coupled['directions_started'], '每核一个false初P')
    require(old.integer(report.get('ipm_iterations'), 200) and coupled['constant_calls_started'] <= report['ipm_iterations'] + 1,
        '已提交HSD步与当前常量尝试保持原前缀')
    if coupled['last'] is not None:
        require(coupled['last']['attempt_iteration'] <= report['ipm_iterations'] + 1, 'last真实尝试至多领先已提交步1')
    public = report.get('ipm_homogeneous_scaled_inexact_returns')
    require(old.integer(public), '原公共inexact字段必须真实非负整数')
    gap = coupled['initial_forcing_returns'] + coupled['candidate_forcing_returns'] - public
    require(0 <= gap <= 1, 'private kernel-return与public完整delivery-return仅可末差1')
    if gap:
        last = coupled['last']
        require(last is not None and last['status'] in {'forced_initial', 'forced_candidate'} and
            last['initial_forcing' if last['status'] == 'forced_initial' else 'candidate_forcing']['decision'] == 'returned' and
            all(coupled[stem + '_started'] == coupled[stem + '_completed'] for stem in
                ('constant_calls', 'directions', 'preconditioner', 'products', 'krylov', 'true_checks')),
            '差1只能最后forced kernel返回后delivery失败，无更晚partial工作')
        require(type(primary) is dict and primary['child']['started'] == primary['child']['completed'] == 1 and
            primary['child']['status'] in {'time_limit', 'numerical_error', 'error'},
            '依真实child失败快照判交付partial，最终原q可恢复通过')
        require(type(energy) is dict and energy['started'] == energy['completed'] == coupled['constant_calls_completed'] and
            energy['last'] is not None and energy['last']['attempt_iteration'] == last['attempt_iteration'],
            '无更晚常量/能量活动；源码和真实末预算组件另核')




# 供ROOT拼接到v98-candidate-audit.py；依赖该文件已有的只读工具，不自行执行。


RENAMES = (
    ('solver_v97_', 'solver_v98_'), ('v97_core', 'v98_core'),
    ('v97_solve_coupled', 'v98_solve_coupled'), ('v97_coupled_enabled', 'v98_coupled_enabled'),
    ('v97_role', 'v98_role'), ('v97_iteration', 'v98_iteration'),
    ('v97_energy_denominator', 'v98_energy_denominator'),
    ('v97_tolerances_and_rhs_norms', 'v98_tolerances_and_rhs_norms'),
    ('v97_deliver_direction', 'v98_deliver_direction'),
    ('V97_PRESOLVE', 'V98_PRESOLVE'), ('V97_COUPLED', 'V98_COUPLED'), ('V97_ENERGY', 'V98_ENERGY'),
    ('v97-presolve-state.hpp', 'v98-presolve-state.hpp'), ('v97-coupled-state.hpp', 'v98-coupled-state.hpp'),
    ('v97-coupled.hpp', 'v98-coupled.hpp'), ('solver::v97_coupled', 'solver::v98_coupled'),
)


def rename98(text):
    for before, after in RENAMES:
        text = text.replace(before, after)
    return text


def source_evidence(work=None):
    """以父封存审计、独立字面增量和实际35锚交叉核，不导入本轮转换器。"""
    import ast
    root = Path(__file__).resolve().parents[1]
    data = root / 'knowledge/experiments/data/2026-10-12-halpern-ipm'
    work = Path(work) if work is not None else root / '.tools/v98-presolve'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    text = lambda p: Path(p).read_text(encoding='utf-8').replace('\r\n', '\n')
    require(sha(data / 'v98-plan.json') == PLAN_SHA256, 'v98冻结计划变化')
    plan = read(data / 'v98-plan.json')
    require(sha(data / 'v97-closeout.json') == plan['evidence_sha256'], '父v97证据变化')
    parent = read(data / 'v97-closeout.json')
    old_path = root / '.tools/v97-candidate-audit.py'
    require(sha(old_path) == parent['independent_auditors'][str(old_path)], '父纯读审计身份变化')
    old = types.ModuleType('v98_locked_v97_source_auditor'); old.__file__ = str(old_path)
    exec(compile(old_path.read_bytes(), str(old_path), 'exec'), old.__dict__)
    old_identity = read(data / 'v97-run-identity.json')
    old_work = Path(old_identity['build_manifest']).parent
    require(sha(data / 'v97-run-identity.json') == parent['run_identity_sha256'] and
        sha(old_work / 'build-manifest.json') == parent['build_manifest_sha256'], '父实际构建来源变化')
    parent_evidence = old.source_evidence(old_work)
    require(parent_evidence == parent['energy_source_evidence'] and parent_evidence['passed'] and parent_evidence['anchors'] == 33,
        '父33锚实际源证据须与冻结结果全等')
    require(sha(data / 'v97-ipm-instrumentation.py') == plan['parent_instrumentation_sha256'] and
        sha(data / 'v97-coupled.hpp') == plan['algorithm_parent_coupled_kernel_sha256'] and
        sha(data / 'v92-coupled.hpp') == plan['parent_coupled_kernel_sha256'] and
        sha(data / 'v93-energy.hpp') == plan['energy_kernel_sha256'], '新构建的固定父源和旧核身份')
    path = data / 'v98-ipm-instrumentation.py'
    values = {}
    for node in ast.parse(text(path)).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'RENAMES', 'WEIGHT_CPP_CHANGES', 'COUPLED_HEADER_CHANGES'}:
                    require(target.id not in values, '字面表不得重复赋值')
                    values[target.id] = ast.literal_eval(node.value)
    require(values.get('RENAMES') == RENAMES and values.get('COUPLED_HEADER_CHANGES') == [] and len(values) == 3,
        '只允许16改名、两CPP新锚和空头增量表')
    extra = values['WEIGHT_CPP_CHANGES']
    require(type(extra) is list and len(extra) == 2 and [x['label'] for x in extra] ==
        ['forcing_weights_adapter', 'forcing_weights_production_call'] and
        all(set(x) == {'label', 'before', 'after'} for x in extra), '新CPP精确两字面锚')
    changes = [{key: item[key] if key == 'label' else rename98(item[key]) for key in ('label', 'before', 'after')}
        for item in read(old_work / 'transforms.json')['changes']] + extra
    require(len(changes) == 35 and changes == read(work / 'transforms.json')['changes'], '实际35锚顺序独立重建')
    original = text(work / 'frozen/src/ipm.cpp'); parent_actual = rename98(text(old_work / 'ipm-presolve.cpp'))
    actual_raw = (work / 'ipm-presolve.cpp').read_bytes(); actual = text(work / 'ipm-presolve.cpp')
    expected = original
    for index, item in enumerate(changes):
        require(type(item['before']) is str and type(item['after']) is str and item['before'] != item['after'] and
            expected.count(item['before']) == 1, '正向唯一字面锚：' + item['label'])
        expected = expected.replace(item['before'], item['after'], 1)
        if index == 32:
            require(expected == parent_actual, '前33锚必须逐字等于允许改名后的父实际源')
    require(expected == actual, '实际CPP只可为35个冻结锚的变换结果')
    restored = actual
    for index in range(34, -1, -1):
        item = changes[index]
        require(restored.count(item['after']) == 1, '逆向唯一锚：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
        if index == 33:
            require(restored == parent_actual, '删除本轮两锚恢复父源全文')
    require(restored == original, '逆恢复完整冻结生产源')
    for stem in ('presolve-state', 'coupled-state'):
        require(text(work / ('v98-' + stem + '.hpp')) == rename98(text(old_work / ('v97-' + stem + '.hpp'))),
            'primary/schema3/Energy状态头保持，仅允许改名')
    kernel_parent = text(data / 'v97-coupled.hpp').replace('solver::v97_coupled', 'solver::v98_coupled')
    kernel = kernel_parent
    require(len(KERNEL_EDITS) == 7, '纯核精确七字面增量')
    for before, after in KERNEL_EDITS:
        require(before != after and kernel.count(before) == 1, '每个核锚须唯一')
        kernel = kernel.replace(before, after, 1)
    require(kernel == text(data / 'v98-coupled.hpp'), '实际纯核仅七项允许增量')
    restored_kernel = kernel
    for before, after in reversed(KERNEL_EDITS):
        require(restored_kernel.count(after) == 1, '每个核逆锚须唯一')
        restored_kernel = restored_kernel.replace(after, before, 1)
    require(restored_kernel == kernel_parent, '七项逆增量恢复所有真实gate/best/restart原文')
    build = read(work / 'build-manifest.json')
    names = {'ipm-presolve.cpp', 'v98-presolve-state.hpp', 'v98-coupled-state.hpp', 'v98-coupled.hpp',
        'v92-coupled.hpp', 'v93-energy.hpp', 'coupled-self.cpp'}
    generated = build['generated_inputs_before_and_after']
    require(set(generated) == {str(work / name) for name in names} and
        all(sha(name) == digest for name, digest in generated.items()), '七实际生成源构建前后身份')
    require(build['protected_before_and_after'][str(path)] == sha(path), 'AST源是实际保护的编译源')
    for name in ('v98-coupled.hpp', 'v92-coupled.hpp', 'v93-energy.hpp'):
        require((work / name).read_bytes() == (data / name).read_bytes(), '真实编译头字节：' + name)
    require((work / 'coupled-self.cpp').read_bytes() == (data / 'v98-coupled-self-test.cpp').read_bytes(), '数学测试真实副本')
    adapter = extra[0]['after']; production = extra[1]['after']
    require(actual.count(adapter) == actual.count(production) == 1 and
        'bool allow_candidates = false,\n    bool forcing_weights = false)' in adapter and
        'multiply, multiply_wide, call.work, budget, &forcing, allow_candidates, forcing_weights)' in adapter and
        'budget, v98_role, v98_iteration, &forced, true, true);' in production,
        'adapter默认false保控制，真实production显式启用两个能力')
    for begin, end in (('V96ToleranceData v98_tolerances_and_rhs_norms(', 'HomogeneousDirection v98_deliver_direction('),
                       ('HomogeneousDirection v94_deliver_direction(', 'long double v94_corrector_rhs(')):
        require(actual[actual.index(begin):actual.index(end, actual.index(begin))] ==
            parent_actual[parent_actual.index(begin):parent_actual.index(end, parent_actual.index(begin))],
            '完整RHS/B3原扫与LD pack末预算保持')
    return dict(passed=True, anchors=35, parent_anchors=33, wide_anchors=8, reversible=True, parent_restorable=True,
        kernel_literal_changes=7, header_literal_changes=0, allowed_renames=16,
        three_search_scales_consistent=True, true_gates_unchanged=True, initial_forcing_preserved=True,
        strict_best_and_restart_preserved=True, rhs_norms_same_scan=True, public_inexact_after_delivery=True,
        candidate_adapter_default_false=True, forcing_weights_adapter_default_false=True,
        production_explicit_two_flags=True, coupled_schema=3, actual_generated_sources=7,
        actual_source_sha256=hashlib.sha256(actual_raw).hexdigest(), instrumentation_sha256=sha(path),
        coupled_kernel_sha256=sha(data / 'v98-coupled.hpp'), algorithm_parent_kernel_sha256=sha(data / 'v97-coupled.hpp'),
        legacy_kernel_sha256=sha(data / 'v92-coupled.hpp'), energy_kernel_sha256=sha(data / 'v93-energy.hpp'),
        frozen_parent_source_evidence=parent_evidence,
        scope='仅核源码身份/尺度一致/旧真实gate及交付合同；新尺度改变真验时点和轨迹，不证明有限精度收敛或LP最优')


KERNEL_EDITS = [
    ('// T是按固定五块门槛组成的对角阵，Arnoldi作用于T^-1 A P T。',
     '// T默认取strict五块门槛；显式forcing_weights时只将搜索尺度换为同一F。\n// 真实strict/forcing验收仍用原门槛；Arnoldi作用于T^-1 A P T。'),
    ('    Work& work, const Budget& check = {}, const InitialForcing* initial = nullptr, bool allow_candidates = false) {',
     '    Work& work, const Budget& check = {}, const InitialForcing* initial = nullptr, bool allow_candidates = false,\n    bool forcing_weights = false) {'),
    ('  if (allow_candidates && !initial)\n    throw std::invalid_argument("候选forcing必须提供原完整右端块范数");',
     '  if (forcing_weights && (!allow_candidates || !initial))\n    throw std::invalid_argument("受控FG搜索尺度要求候选forcing与原完整右端块范数");\n  if (allow_candidates && !initial)\n    throw std::invalid_argument("候选forcing必须提供原完整右端块范数");'),
    ('  for (int cycle = 0; cycle < max_cycles; ++cycle) {',
     '''  std::array<long double, 5> search_scales{};
  for (std::size_t block = 0; block < 5; ++block) {
    search_scales[block] = tolerances[block];
    if (forcing_weights && block < 3)
      search_scales[block] = std::max(search_scales[block],
          0.1L * static_cast<long double>(work.initial_forcing->rhs_norms[block]));
  }
  for (int cycle = 0; cycle < max_cycles; ++cycle) {'''),
    ('best_assessment.residual[i] / tolerances[layout.block(i)]',
     'best_assessment.residual[i] / search_scales[layout.block(i)]'),
    ('static_cast<long double>(physical_rhs[i]) * tolerances[layout.block(i)]',
     'static_cast<long double>(physical_rhs[i]) * search_scales[layout.block(i)]'),
    ('image[i] / tolerances[layout.block(i)]', 'image[i] / search_scales[layout.block(i)]'),
]
