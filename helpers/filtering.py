def filter_problematic_clusters(df):

    def strategy_based_on_manual_identification(row):
        if row['journal'] == 'avenirgdl' and row['page'] == 4:
            return False
        if row['journal'] == 'courriergdl' and row['page'] == 4 and 1860 <= row['year'] <= 1870:
            return False
        if row['journal'] == 'lunion' and row['page'] in {1, 4}:
            return False
        if row['journal'] == 'luxwort' and row['page'] == 4 and 1860 <= row['year'] <= 1890:
            return False
        if row['journal'] == 'luxzeit' and row['page'] == 4:
            return False
        if row['journal'] == 'volkfreu1869' and row['page'] == 4:
            return False
        if row['journal'] == 'waechtersauer' and row['page'] == 1 and 1862 <= row['year']:
            return False
        if row['journal'] == 'diekwochen' and row['page'] == 4:
            return False
        if row['journal'] == 'gazgrdlux' and row['page'] == 4:
            return False
        return True

    return df[df.apply(strategy_based_on_manual_identification, axis=1)]
