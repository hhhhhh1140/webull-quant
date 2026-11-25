import pytest

from wquant.scalp import ScalpParams
from wquant.scalp_report import metrics, print_report, replay


def test_results_carry_cash_and_charge_costs_once(monkeypatch, capsys):
    seen = []

    def session(long_bars, short_bars, params, close):
        seen.append((params.capital, params.fee_per_order))
        return [{"pnl": -10, "costs": 2}], {"qty": 0}

    monkeypatch.setattr("wquant.scalp_report.run_session", session)
    days = [(i, None, None, None) for i in range(4)]
    params = ScalpParams(capital=3000, fee_per_order=0.10)
    stats = metrics(replay(days, params))
    assert [capital for capital, _ in seen] == [3000, 2990, 2980, 2970]
    assert stats["毛利$"] == -32
    assert stats["净利$"] == -40
    assert stats["已平仓回撤"] == pytest.approx(-40 / 3000)
    seen.clear()
    print_report(days, params)
    assert seen[-2:] == [(2980, 0.20), (2970, 0.20)]
    assert "后段验证" in capsys.readouterr().out
