import json
from typing import Optional, TypeVar, Callable, Optional
import os
import pickle
import warnings
import pandas as pd
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *
from alphagen_qlib.calculator import QLibStockDataCalculator

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_recent_data, load_alpha_pool_by_path

_T = TypeVar("_T")


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float

    def to_dict(self):
        return {
            "sharpe": getattr(self, "sharpe", None),  # If attribute doesn't exist, return None
            "annual_return": getattr(self, "annual_return", None),
            "max_drawdown": getattr(self, "max_drawdown", None),
            "information_ratio": getattr(self, "information_ratio", None),
            "annual_excess_return": getattr(self, "annual_excess_return", None),
            "excess_max_drawdown": getattr(self, "excess_max_drawdown", None),
        }


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = None,
        return_report: bool = False
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=1,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]
        if output_prefix is not None:
            report.to_csv(output_prefix + "-report.csv", index=False)
            # dump_pickle(output_prefix + "-report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "-graph.pkl", lambda: graph, True)
            write_all_text(output_prefix + "-result.json", json.dumps(result.to_dict(), ensure_ascii=False, indent=4))

        print(report)
        print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )


def fun1():
    qlib_backtest = QlibBacktest()
    data = StockData(instrument='csi300',
                     start_time='2021-01-01',
                     end_time='2022-12-31')
    # 因子组合
    POOL_PATH = '/home/kk/project/alphagen/checkpoint/new_all_5_2_20241202160424/100352_steps_pool.json'
    # data, latest_date = load_recent_data(instrument='csi300', window_size=365, offset=1)
    # pd.DataFrame(data.data.cpu().numpy()).to_csv('/home/kk/project/alphagen/file/data_cpu.csv')
    calculator = QLibStockDataCalculator(data=data, target=None)
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)
    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    # ensemble_alpha_cpu = ensemble_alpha.cpu()
    # pd.DataFrame(ensemble_alpha_cpu.numpy()).to_csv('/home/kk/project/alphagen/file/ensemble_alpha.csv')
    df = data.make_dataframe(ensemble_alpha)
    # 单个因子
    # expr = Mul(EMA(Sub(Delta(Mul(Log(open_),Constant(-30.0)),50),Constant(-0.01)),40),Mul(Div(Abs(EMA(low,50)),close),Constant(0.01)))
    # expr = Mul(Abs(vwap),Div(Constant(-30.0),Mul(Mul(close,Constant(-0.01)),Constant(-0.5))))
    # data_df = data.make_dataframe(expr.evaluate(data))
    output_prefix = '/home/kk/project/alphagen/file/20241223'
    qlib_backtest.run(df, output_prefix=output_prefix)

def fun2():
    # # 加载回测报告 (report.pkl)
    # with open('/home/kk/project/alphagen/file/20241223-report.pkl', 'rb') as f:
    #     report = pickle.load(f)
    #     print(report)  # 打印报告内容
    #
    # 加载回测图表 (graph.pkl)
    # with open('/home/kk/project/alphagen/file/20241223-graph.pkl', 'rb') as f:
    #     graph = pickle.load(f)
    #     print(graph)  # 打印图表数据，具体如何显示取决于图表的类型
    with open('/home/kk/project/alphagen/file/20241223-graph.pkl', 'rb') as f:
        graph = pickle.load(f)
        # 假设 graph 是一个 matplotlib 图形对象
        graph.show()

if __name__ == "__main__":
    fun1()
    # fun2()



