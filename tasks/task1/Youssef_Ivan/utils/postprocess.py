import pandas as pd
from typing import List

def save_xlsx(df_list: List[pd.DataFrame], names_list = List[str]):

    #writer = pd.ExcelWriter('pos_full_pred_splitted.xlsx')

    print("WTF")
    with pd.ExcelWriter('pos_full_pred_splitted.xlsx') as writer:


        for df, name in zip(df_list, names_list):

            df.to_excel(writer, sheet_name=name)

        writer.save()

    return True

    



