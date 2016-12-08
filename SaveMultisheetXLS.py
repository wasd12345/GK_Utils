from pandas import ExcelWriter


def SaveMultisheetXLS(list_dfs, list_sheetnames, xls_path):
    """
    xls_path - must be .xls or else will not save sheets properly
    """
    writer = ExcelWriter(xls_path)
    for s in xrange(len(list_dfs)):
        list_dfs[s].to_excel(writer,list_sheetnames[s],index=False)
    writer.save()