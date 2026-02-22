import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)

input_file = "LLCP2023_subset.csv"

target_column = 'DIABETE4'

def train_and_evaluate():
    predictors = [
        '_RFHYPE6',
        'TOLDHI3',
        '_BMI5',
        'SMOKE100',
        'CVDSTRK3',
        '_MICHD',
        '_TOTINDA',
        '_HLTHPL1',
        'MEDCOST1',
        'GENHLTH',
        'MENTHLTH',
        'PHYSHLTH',
        'DIFFWALK',
        'SEXVAR',
        '_AGEG5YR',
        'EDUCA',
        'INCOME3',
    ]

    numeric = [
        '_BMI5',
        'MENTHLTH',
        'PHYSHLTH'
    ]

    df = pd.read_csv(input_file)

    df_clean = df[df[target_column].isin([1.0, 3.0])].copy()
    df_clean['TARGET_DIABETES'] = df_clean[target_column].map({1.0: 1, 3.0: 0})

    existing_features = [col for col in predictors if col in df_clean.columns]
    columns_to_keep = [target_column, 'TARGET_DIABETES'] + existing_features
    df_processed = df_clean[columns_to_keep].copy()

    missing_codes = [9.0, 14.0, 77.0, 99.0, 777.0, 999.0, 9999.0, 99900.0]

    for col in existing_features:
        df_processed[col] = df_processed[col].replace(missing_codes, np.nan)


    for col in ['MENTHLTH', 'PHYSHLTH']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace({88.0: 0.0})


    before_cleaning = len(df_processed)
    df_processed = df_processed.dropna()
    after_cleaning = len(df_processed)

    print(f"\nLiczba danych przed oczyszczaniem: {before_cleaning}")
    print(f"Liczba wierszy po oczyszczeniu zbioru: {after_cleaning}")

    summary_df = pd.concat([
        df_processed['TARGET_DIABETES'].value_counts(),
        df_processed['TARGET_DIABETES'].value_counts(normalize=True).mul(100).round(2)
    ], axis=1, keys=['Liczebność', 'Udział procentowy'])

    print("\nRozkład zmiennej docelowej:")
    print(summary_df)


    print(f"\nTypy danych")
    df_processed.info(verbose=True, show_counts=True)

    print("\nStatystyki Opisowe dla zmiennych numerycznych:")
    numeric_cols = [col for col in numeric if col in df_processed.columns]
    print(df_processed[numeric_cols].describe())

    existing_numeric_cols = [col for col in numeric if col in existing_features]
    existing_categorial_cols = [col for col in existing_features if col not in existing_numeric_cols]

    y = df_processed['TARGET_DIABETES']
    X_df = df_processed[existing_numeric_cols + existing_categorial_cols]
    X_encoded = pd.get_dummies(X_df, columns=existing_categorial_cols, drop_first=False, dtype=int)

    final_model_columns = X_encoded.columns.tolist()
    joblib.dump(final_model_columns, 'model_columns.joblib')

    print("\nPodział na zbiór treningowy i testowy")
    X = X_encoded
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
    print(f"Zbiór testowy: {X_test.shape[0]} próbek")


    scaler = StandardScaler()
    X_train.loc[:, existing_numeric_cols] = scaler.fit_transform(X_train[existing_numeric_cols])
    X_test.loc[:, existing_numeric_cols] = scaler.transform(X_test[existing_numeric_cols])

    joblib.dump(scaler, 'diabetes_scaler.joblib')

    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )

    model.fit(X_train, y_train)

    joblib.dump(model, 'diabetes_model.joblib')

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    print("\nMacierz Pomyłek:")
    print(cm)


    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Brak Cukrzycy (0)', 'Cukrzyca (1)'],
        yticklabels=['Brak Cukrzycy (0)', 'Cukrzyca (1)']
    )
    plt.xlabel('Przewidziana klasa', fontsize=12)
    plt.ylabel('Prawdziwa klasa', fontsize=12)
    plt.show()

    print("\nRaport Klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=['Brak Cukrzycy (0)', 'Cukrzyca (1)']))

    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC: {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(10, 7))
    plt.xticks( fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Model losowy (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specyficzność)', fontsize=14)
    plt.ylabel('True Positive Rate (Czułość)', fontsize=14)
    plt.legend(loc="lower right", fontsize=16)
    plt.grid(alpha=1)

    plt.show()



    print(f"\n\n\nSTATYSTYKA OPISOWA I ANALIZA DANYCH")

    sns.set(style="whitegrid")

    df_processed['TARGET_DIABETES_label'] = df_processed['TARGET_DIABETES'].map(
        {1: 'Cukrzyca (1)', 0: 'Brak Cukrzycy (0)'})

    if '_BMI5' in df_processed.columns:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))

        plot_df = df_processed.copy()
        plot_df['_BMI5'] = plot_df['_BMI5'] / 100
        plot_df = plot_df[(plot_df['_BMI5'] > 15) & (plot_df['_BMI5'] < 65)]

        ax = sns.histplot(
            data=plot_df,
            x='_BMI5',
            hue='TARGET_DIABETES',
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
            binwidth=1,
            alpha=0.4,
            palette=['#3498db', '#e67e22'],
            legend=False
        )
        plt.xticks(range(15, 70, 5), fontsize=16)
        plt.yticks(fontsize=16)

        plt.axvline(x=25, color='#27ae60', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.axvline(x=30, color='#c0392b', linestyle='--', linewidth=2.5, alpha=0.8)

        plt.xlabel('Wskaźnik BMI (kg/m²)', fontsize=16, weight='bold', labelpad=15)
        plt.ylabel('Gęstość prawdopodobieństwa', fontsize=16, weight='bold', labelpad=15)

        blue_patch = mpatches.Patch(color='#3498db', alpha=0.6, label='Brak Cukrzycy')
        orange_patch = mpatches.Patch(color='#e67e22', alpha=0.6, label='Cukrzyca')
        line_overweight = mlines.Line2D([], [], color='#27ae60', linestyle='--',
                                        linewidth=2, label='Nadwaga (BMI ≥ 25)')
        line_obesity = mlines.Line2D([], [], color='#c0392b', linestyle='--',
                                     linewidth=2, label='Otyłość (BMI ≥ 30)')

        plt.legend(handles=[blue_patch, orange_patch, line_overweight, line_obesity],
                   title='Kategorie i progi diagnostyczne',
                   title_fontsize=16,
                   fontsize=15,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=2,
                   frameon=False)

        plt.xlim(15, 65)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    if 'GENHLTH' in df_processed.columns:
        sns.set_theme(style="whitegrid")
        plot_df = df_processed[df_processed['GENHLTH'].isin([1, 2, 3, 4, 5])].copy()
        props = pd.crosstab(plot_df['GENHLTH'], plot_df['TARGET_DIABETES'], normalize='index')

        health_labels = {
            1: 'Doskonałe',
            2: 'Bardzo dobre',
            3: 'Dobre',
            4: 'Średnie',
            5: 'Złe'
        }
        props.index = props.index.map(health_labels)
        ax = props.plot(
            kind='bar',
            stacked=True,
            figsize=(14, 10),
            color=['#3498db', '#e67e22'],
            width=0.8,
            edgecolor='white'
        )

        for c in ax.containers:
            labels = [f'{v.get_height() * 100:.1f}%' if v.get_height() > 0.02 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white',
                         fontweight='bold', fontsize=18)

        plt.xlabel('Ogólna, subiektywna ocena zdrowia', fontsize=18, weight='bold', labelpad=15)
        plt.ylabel('Struktura grupy (%)', fontsize=18, weight='bold', labelpad=15)

        plt.xticks(rotation=0, fontsize=18)
        plt.yticks(fontsize=18)

        plt.legend(title='Status Cukrzycy',
                   labels=['Brak Cukrzycy', 'Cukrzyca'],
                   loc='lower center',
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=2,
                   frameon=False,
                   fontsize=18,
                   title_fontsize=18)

        plt.grid(axis='y', linestyle='--', alpha=1)
        plt.tight_layout()
        plt.show()

    if '_AGEG5YR' in df_processed.columns:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(14, 9))

        age_map = {
            1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
            6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
            11: '70-74', 12: '75-79', 13: '80+'
        }

        age_order = sorted([x for x in df_processed['_AGEG5YR'].unique() if x in age_map])

        ax = sns.countplot(
            data=df_processed,
            x='_AGEG5YR',
            hue='TARGET_DIABETES',
            order=age_order,
            palette=['#3498db', '#e67e22'],
            edgecolor='black',
            linewidth=0.9
        )

        new_labels = [age_map.get(x, str(x)) for x in age_order]
        ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=19)

        plt.xlabel('Przedział wiekowy (lata)', fontsize=17, weight='bold', labelpad=15)
        plt.ylabel('Liczba badanych', fontsize=17, weight='bold', labelpad=15)
        plt.yticks(fontsize=19)

        plt.legend(title='Status Cukrzycy',
                   labels=['Brak Cukrzycy', 'Cukrzyca'],
                   fontsize=15,
                   title_fontsize=18,
                   loc='lower center',
                   bbox_to_anchor=(0.5, 1.02),
                   ncol=2,
                   frameon=False)

        ax.yaxis.grid(True, linestyle='--', alpha=0.8)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.show()

        sns.set_theme(style="white")
        plt.figure(figsize=(16, 12))

        expanded_numeric_cols = numeric_cols + ['_AGEG5YR', 'EDUCA', 'INCOME3', 'GENHLTH', 'TARGET_DIABETES']
        cols_to_plot = [col for col in expanded_numeric_cols if col in df_processed.columns]
        corr_matrix = df_processed[cols_to_plot].corr(method='spearman')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='RdBu_r',
            vmax=1, vmin=-1,
            center=0,
            square=True,
            linewidths=1.5,
            linecolor='#d1d1d1',
            cbar_kws={"shrink": .8},
            annot_kws={"size": 14, "weight": "bold"}
        )

        plt.xticks(rotation=45, ha='right', fontsize=18, weight='bold')
        plt.yticks(rotation=0, fontsize=18, weight='bold')

        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.show()
        plt.show()

    print("\nŚrednie wartości cech w podziale na status cukrzycy:")
    stats_grouped = df_processed.groupby('TARGET_DIABETES')[numeric_cols].mean()
    print(stats_grouped)

    sns.set_theme(style="whitegrid")

    # Dane
    target_counts = df_processed['TARGET_DIABETES'].value_counts()
    counts = [target_counts[0], target_counts[1]]
    labels = ['Brak Cukrzycy', 'Cukrzyca']

    colors = ['#3498db', '#e67e22']
    explode = (0.05, 0.05)

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 2}
    )

    plt.setp(autotexts, size=16, weight="bold", color="white")

    ax.text(0, 0, f'Suma próbek\n{sum(counts)}', ha='center', va='center',
            fontsize=16, weight='bold', color='#333333')

    legend_labels = [f'{l}: {c}' for l, c in zip(labels, counts)]
    ax.legend(wedges, legend_labels,
              title="Klasy i liczebność",
              loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=2,
              fontsize=16,
              title_fontsize=16,
              frameon=False)


    plt.tight_layout()
    plt.show()

    coefficients = model.coef_[0]
    feature_names = X_train.columns

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })


    feature_importance['Odds_Ratio'] = np.exp(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 18))

    colors = ['#d62728' if c > 0 else '#2ca02c' for c in feature_importance['Coefficient']]

    ax = sns.barplot(
        x='Coefficient',
        y='Feature',
        data=feature_importance,
        palette=colors,
        edgecolor='black',
        linewidth=0.8
    )

    for i, (coef, or_val) in enumerate(zip(feature_importance['Coefficient'], feature_importance['Odds_Ratio'])):
        x_pos = coef + 0.02 if coef > 0 else coef - 0.02
        ha = 'left' if coef > 0 else 'right'

        ax.text(
            x_pos, i,
            f'OR: {or_val:.2f}',
            va='center',
            ha=ha,
            fontsize=14,
            fontweight='bold',
            color='#333333'
        )


    plt.xlabel('Wartość współczynnika (Log-Odds)', fontsize=18, weight='bold', labelpad=15)
    plt.ylabel('Cecha / Zmienna', fontsize=18, weight='bold', labelpad=15)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=1)

    limit = max(abs(feature_importance['Coefficient'].min()), abs(feature_importance['Coefficient'].max())) + 0.3
    plt.xlim(-limit, limit)

    plt.tight_layout()
    plt.show()

    print("\n--- Generowanie wykresu dla danych PRZED czyszczeniem ---")

    # Używamy df_clean, który zawiera dane przed usunięciem NaN
    target_counts_raw = df_clean['TARGET_DIABETES'].value_counts()

    # Upewniamy się, że kolejność jest zgodna (0: Brak, 1: Cukrzyca)
    # Jeśli w surowych danych są braki w tej kolumnie, value_counts je pomija domyślnie
    counts_raw = [target_counts_raw.get(0, 0), target_counts_raw.get(1, 0)]
    labels = ['Brak Cukrzycy', 'Cukrzyca']

    colors = ['#3498db', '#e67e22']
    explode = (0.05, 0.05)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        counts_raw,
        labels=None,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        pctdistance=0.75,
        wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 2}
    )

    plt.setp(autotexts, size=16, weight="bold", color="white")

    # Dodano dopisek "(Surowe)" w środku, aby odróżnić wykresy
    ax.text(0, 0, f'Suma próbek\n{sum(counts_raw)}', ha='center', va='center',
            fontsize=16, weight='bold', color='#333333')

    legend_labels = [f'{l}: {c}' for l, c in zip(labels, counts_raw)]
    ax.legend(wedges, legend_labels,
              title="Klasy i liczebność",
              loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=2,
              fontsize=16,
              title_fontsize=16,
              frameon=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()