import numpy as np
import pandas as pd
from lib.data_utils import CACHE_DIR, load_results
from lib.semi_supervised_utils import PROPORTIONS
from collections import namedtuple

ReportSection = namedtuple('ReportSection', ['title', 'dataframe', 'insights'])

def get_mean_f1(results, proportion, metric='macro avg'):
    scores = [run[metric]['f1-score'] for run in results.get(str(proportion), [])]
    return np.mean(scores) if scores else 0.0

def analyze_f1_tables(summary_df, baseline_f1):
    df = summary_df[['proportion', 'st_f1', 'lp_f1']].copy()
    df['baseline'] = baseline_f1
    df.columns = ['Proportion', 'Self-Training', 'Label Propagation', 'Baseline']
    df['Proportion'] = df['Proportion'].map('{:.1%}'.format)
    for col in ['Self-Training', 'Label Propagation', 'Baseline']:
        df[col] = df[col].map('{:.2f}'.format)
    return ReportSection("--- F1 Scores ---", df, [])

def analyze_win_rate(summary_df):
    wins = summary_df['winner'].value_counts().to_dict()
    lp_wins = wins.get('LP', 0)
    st_wins = wins.get('ST', 0)
    total = lp_wins + st_wins
    assert total > 0

    insights = []
    lp_win_rate = lp_wins / total * 100
    insights.append(f"LP wins {lp_wins}/{total} times ({lp_win_rate:.1f}%)")
        
    return ReportSection("--- F1 Win rate ---", None, insights)


def analyze_class8_improvement(summary_df):
    p_low = summary_df['proportion'].min()
    p_high = summary_df['proportion'].max()
    
    low_row = summary_df.query(f"proportion == {p_low}").iloc[0]
    high_row = summary_df.query(f"proportion == {p_high}").iloc[0]

    lp_improvement = high_row.lp_f1_c8 - low_row.lp_f1_c8
    st_improvement = high_row.st_f1_c8 - low_row.st_f1_c8
    
    class8_data = [
        {
            "Proportion": f"{p_low:.1%}",
            "ST F1": f"{low_row.st_f1_c8:.3f}",
            "LP F1": f"{low_row.lp_f1_c8:.3f}",
        },
        {
            "Proportion": f"{p_high:.1%}",
            "ST F1": f"{high_row.st_f1_c8:.3f}",
            "LP F1": f"{high_row.lp_f1_c8:.3f}",
        },
    ]

    insights = [
        f"• LP improves by {lp_improvement:.3f} F1 points from {p_low:.1%} to {p_high:.1%}",
        f"• ST improves by {st_improvement:.3f} F1 points from {p_low:.1%} to {p_high:.1%}",
    ]
    
    return ReportSection(
        "--- Class 8 Improvement ---", pd.DataFrame(class8_data), insights
    )

def print_report(report_sections):
    for section in report_sections:
        print(f"\n{section.title}")
        if section.dataframe is not None:
            print(section.dataframe.to_string(index=False))
        if section.insights:
            print()
            print("Key Insights:")
            for insight in section.insights:
                print(insight)

def main():
    baseline_results = load_results(CACHE_DIR / "results" / "baseline.json")
    st_results = load_results(CACHE_DIR / "results" / "self_training.json")
    lp_results = load_results(CACHE_DIR / "results" / "label_propagation.json")
    baseline_f1 = baseline_results['report']['macro avg']['f1-score']
    
    print(f"Baseline SVM Macro F1: {baseline_f1:.2f}")

    data = []
    for p in PROPORTIONS:
        data.append({
            'proportion': p,
            'st_f1': get_mean_f1(st_results, p),
            'lp_f1': get_mean_f1(lp_results, p),
            'st_f1_c8': get_mean_f1(st_results, p, '8'),
            'lp_f1_c8': get_mean_f1(lp_results, p, '8'),
        })
    df = pd.DataFrame(data)
    df['winner'] = np.where(df['lp_f1'] > df['st_f1'], 'LP', 'ST')
    df['lp_advantage'] = df['lp_f1'] - df['st_f1']
    df['lp_ratio'] = df['lp_f1'] / df['st_f1'].replace(0, np.inf)
    summary_df = df
    
    report_sections = [
        analyze_f1_tables(summary_df, baseline_f1),
        analyze_win_rate(summary_df),
        analyze_class8_improvement(summary_df),
    ]

    print_report(report_sections)


if __name__ == "__main__":
    main()