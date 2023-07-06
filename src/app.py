import pandas as pd
from dash import Dash, html
from dash import dcc
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto # pip install dash-cytoscape
import numpy as np

import visdcc # pip install visdcc
# In[]
origin_key_dict_pd = pd.read_csv('../../new_data/entityDict.csv')[["label", "keywords"]]#種類、名稱

keyword_class_list = ["com", "rocket", "org", "satellite", "term", "loc"]
Sen_Doc_list = ["Sentence", "Document"]
# In[]
lemma = pd.read_csv('../../new_data/doc_label_table.csv')
X = pd.read_csv('../../new_data/ner_data_bio.csv')
#raw_S = pd.read_csv('./new_data/sen_raw_data.csv')
XX_Sent = pd.read_csv('../../new_data/SenDTM.csv')
XX_Doc = pd.read_csv('../../new_data/DocDTM.csv')
senlabel = pd.read_csv('../../new_data/sen_label_table.csv')
raw_S = pd.read_csv('../../new_data/sen_raw_data.csv')
coo_df = pd.read_csv('../../new_data/DocCO_format.csv')
CR_doc = pd.read_csv('../../new_data/DocCR.csv')#CR_doc2 = pd.read_csv('./data/CRdoc1224.csv',index_col=0)
CR_sen = pd.read_csv('../../new_data/SenCR.csv')
CO_doc = pd.read_csv('../../new_data/DocCO.csv')
CO_sen = pd.read_csv('../../new_data/DocCO.csv')
# In[]
color_list = ['#FB8072','#80B1D3','#BFB39B','#FDB462','#B3DE69','#FFFFB3']
color_dict = dict(zip(keyword_class_list, color_list))
#Z = "Carnegie_Mellon_University"
#Z = "GPS"
#Z = Z.lower()
def get_element_modify(Unit, Z, type, total_nodes_num, threshold):
            
        if type == 'correlation':
            if Unit == "Document":
                input_data = CR_doc

            elif Unit == "Sentence":
                input_data = CR_sen
            
            v = input_data[Z].tolist()
            v = list(enumerate(v))
            v = sorted(v, key=lambda x: x[1], reverse=True)
            v_index = [i for i, _ in v][:total_nodes_num]
            col_index = [((input_data.columns).tolist())[i] for i in v_index]
            x = input_data.loc[v_index, col_index]
            x.columns = v_index
            #x = (input_data.loc[v_index, col_index]).set_index(pd.Index(col_index))
            
            melted_df = x.stack().reset_index()
            melted_df.columns = ['from', 'to', 'Value']
            melted_df = melted_df[melted_df['Value'] != 1].reset_index(drop=True)
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)
            
            #melted_df['drop_subset'] = melted_df['from'].astype(str) + "_" + melted_df['to'].astype(str)
            
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, threshold)
            
            #melted_df_thres = melted_df[melted_df['Value'] > threshold].reset_index(drop=True)
            melted_df_thres = melted_df[melted_df['Value'] > percentile].reset_index(drop=True)
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)
            #melted_df_thres[['from_name', 'to_name']] = np.sort(melted_df_thres[['from_name', 'to_name']], axis=1)
            #melted_df_thres = melted_df_thres.drop_duplicates(subset=['from_name', 'to_name']).reset_index(drop=True)
            
            #melted_df_thres['Unit'] = str(Unit)
            #melted_df_thres['type'] = str(type)
            #melted_df_thres['total_nodes_num'] = str(total_nodes_num)
            #melted_df_thres['threshold'] = str(threshold)
            
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))
                  
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))

        
        elif type == 'co-occurrence':
            if Unit == "Document":
                input_data = CO_doc
                choose_data = CR_doc
                
            elif Unit == "Sentence":
                input_data = CO_sen
                choose_data = CR_sen
                
                
            v = choose_data[Z].tolist()
            v = list(enumerate(v))
            v = sorted(v, key=lambda x: x[1], reverse=True)
            v_index = [i for i, _ in v][:total_nodes_num]
            col_index = [((input_data.columns).tolist())[i] for i in v_index]
            x = input_data.loc[v_index, col_index]
            x.columns = v_index
            #x = (input_data.loc[v_index, col_index]).set_index(pd.Index(col_index))
            
            melted_df = x.stack().reset_index()
            melted_df.columns = ['from', 'to', 'Value']
            melted_df = melted_df[melted_df['Value'] != 1].reset_index(drop=True)
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)
            
            #melted_df['drop_subset'] = melted_df['from'].astype(str) + "_" + melted_df['to'].astype(str)
            
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, threshold)
            
            #melted_df_thres = melted_df[melted_df['Value'] > threshold].reset_index(drop=True)
            melted_df_thres = melted_df[melted_df['Value'] > percentile].reset_index(drop=True)
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)
            #melted_df_thres[['from_name', 'to_name']] = np.sort(melted_df_thres[['from_name', 'to_name']], axis=1)
            #melted_df_thres = melted_df_thres.drop_duplicates(subset=['from_name', 'to_name']).reset_index(drop=True)
            
            #melted_df_thres['Unit'] = str(Unit)
            #melted_df_thres['type'] = str(type)
            #melted_df_thres['total_nodes_num'] = str(total_nodes_num)
            #melted_df_thres['threshold'] = str(threshold)
            
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))
                  
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))

        nodes = [{'id': node, 
                  'label': node, 
                  'group': origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],
                  'color': color_dict.get((origin_key_dict_pd[origin_key_dict_pd["keywords"] == node])["label"].values[0])} for node in nodes_list]
                  #'color': color_list[keyword_class_list.index((origin_key_dict_pd[origin_key_dict_pd["keywords"] == node])["label"].values[0])]} for node in nodes_list]
                  #'color': color_list[(keyword_class_list.index(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1]))]} for node in nodes_list]
        # nodes = [{'id': node, 'label': node} for node in nodes_list]
        #length 篩選edge_df中item1、item2存在於nodes_list，type大於threshold
        edges = [{'id' : row['from_name']+'_'+row['to_name'], 
                  'from': row['from_name'], 'to': row['to_name'], 
                  'classes': type, #cor or coo
                  'weight': row['Value'], 
                  'width':row['edge_width']*6, 
                  'length':row['edge_width']*30} for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) & 
                                                                      melted_df_thres['to_name'].isin(nodes_list))][melted_df_thres['Value'] > threshold].iterrows()]

        info = { "Unit": str(Unit),
                 "type": str(type),
                 "total_nodes_num": total_nodes_num,
                 "threshold": threshold
            }                                                                          
                                                                                  
        data ={'nodes':nodes,
               'edges':edges,
               'info': info
           }
        
        return data
                                
# In[]
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css']

app = Dash(__name__, external_stylesheets = external_stylesheets)

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server


styles = {
    'pre': {
        'border': 'thin lightgrey solid',#邊框屬性
        'overflowX': 'scroll'#內容水平延伸
    }
}
merged_df = pd.DataFrame()
table_data = {
    'dataSource':[],
    'columns':[{'title': 'Date',
                'dataIndex': 'Date',
                'key': 'Date',
                'width': '15%'},
               {'title': 'doc_id',
                'dataIndex': 'id',
                'key': 'id',
                'width': '20%'},
                {'title': 'Recent',
                'dataIndex': 'Recent',
                'key': 'Recent',
                'width': '50%'},
                {'title': 'url',
                'dataIndex': 'url',
                'key': 'url',
                'width': '15%'}],
    
}

app.layout = html.Div([
    html.Div([
        html.P("國防 太空文集 NER單中心網路分析", style={'font-size': '30px'}),
        ## 切換句子或篇下拉式選單
        dcc.Dropdown(
            id='dropdown_choose_SenorDoc',
            value= "Document",
            clearable=False,
            options=[
                {'label': method, 'value': method}
                for i, method in enumerate(Sen_Doc_list)
            ]
        ),
        
        ## 切換 class 下拉式選單
        dcc.Dropdown(
            id='dropdown_choose_class',
            value= 4,
            clearable=False,
            options=[
                {'label': clas, 'value': i}
                for i, clas in enumerate(keyword_class_list)
            ]
        ),
        ## 選擇中心詞
        dcc.Dropdown(
            id='dropdown_choose_name',
            value= 'GPS',
            clearable=False,
            options=[
                    {'label': name, 'value': name}
                    for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[0]]['keywords'].to_list()
            ]
        ),
        dcc.Dropdown(
            id='dropdown-update-layout',
            value='correlation',
            clearable=False,
            options=[
                {'label': name.capitalize(), 'value': name}
                for name in ['correlation', 'co-occurrence']
            ]
        ),
        dcc.Slider(
                id="total_nodes_num_slider", min=4, max=20,step=1,
                marks={i: str(i) for i in range(21)},
                value=4
                #id="total_nodes_num_slider", min=0, max=20,step=1,
                #marks={i: str(i) for i in range(21)},
                #value=7
        ),
        dcc.Slider(
                id="threshold_slide", min=0, max=1,step=0.01,
                marks={i/10: str(i/10) for i in range(11)},
                value=0.01
        ),
        dcc.Markdown('''
                ![Legend](https://i.ibb.co/z2JR9bS/Legend4.png)
        '''),
        # html.Div(id='cytoscape-tapNodeData-output'),
        # html.Div(id='cytoscape-tapEdgeData-output'),
    ],style = {'height' : '100%' ,'width': '20%', 'display': 'inline-block'}),
    html.Div([
            visdcc.Network(
                id='net',
                selection={'nodes': [], 'edges': []},
                options={
                    'autoResize': True,
                    'height': '800px',
                    'width': '100%',
                    'layout': {
                        'randomSeed': 2,
                        'improvedLayout': True,
                        },
                    'physics':{
                         'enabled': True,
                         'barnesHut': {
                                  'theta': 0.5,
                                  'gravitationalConstant': -20000,#repulsion強度
                                  'centralGravity': 0.3,
                                  'springLength': 95,
                                  'springConstant': 0.04,
                                  'damping': 0.09,
                                  #'avoidOverlap': 0.01
                                },
                },
                    'adaptiveTimestep': True,
                    'nodes': {
                        #'x': [100, 200, 300],  # 指定節點的横向位置
                        #'y': [100, 200, 300], 
                        'size': 15  # 调整節點的大小
                        }
                }
            

                        ),
                    
    ],style = {'height' : '100%' ,'width': '50%', 'display': 'inline-block'}),
    
    html.Div([
        #文本元件
        dcc.Textarea(
            id='textarea-example',
            value='paragraph',
            style={'width': '100%', 'height': 300},
            disabled = True,
        ),
        html.Div([
            visdcc.DataTable(
                id         = 'table' ,
                box_type   = 'radio',
                style={'width': '100%', 'height': 500},
                data       = table_data
            )
        ])
        # html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
    ],style = {'height' : '100%' ,'width': '30%', 'display': 'inline-block'}),
    

],style = {'height' : '100%' , 'display': 'flex'})
## 切換 class 下拉式選單
@app.callback(
    Output("dropdown_choose_name", 'options'),
    Input("dropdown_choose_class", "value"),
)
def update_elements(class_idx):#當dropdown_choose_class下拉選單的值發生變化時，會觸發，class_idx類別索引
    ## 選擇中心詞
    options=[
                {'label': name, 'value': name}
                for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[class_idx]]['keywords'].to_list()
            ]
    
    return options

@app.callback(
    Output("threshold_slide", 'min'),
    Output("threshold_slide", 'max'),
    # Output("threshold_slide", 'step'),
    Output("threshold_slide", 'marks'),
    Output("threshold_slide", 'value'),
    Input("dropdown-update-layout", 'value')
)
def update_elements(type):#當dropdown-update-layout下拉選單的值發生變化時，會觸發
    # if type == 'correlation':
    min=0
    max=1
    # step = None
    marks={i/10: str(i/10) for i in range(11)}
    value=0.01
        
    if type == 'co-occurrence':
        min=0
        max=1
        marks={i: str(i) for i in range(0, 1, 0.01)}
        value=0.5
        #max=200
        #marks={i: str(i) for i in range(0, 201, 50)}
        #value=100
    return min, max, marks, value
@app.callback(
    Output("net", 'data'),
    Input('dropdown_choose_SenorDoc', 'value'),
    Input("dropdown_choose_name", 'value'),
    Input("total_nodes_num_slider", "value"),
    Input('dropdown-update-layout', 'value'),
    Input('threshold_slide', 'value'),
    
)
def update_elements(Unit,center_node, total_nodes_num, type, threshold):
    
    return get_element_modify(Unit,center_node, type, total_nodes_num, threshold)

# In[]
#data = 'GPS'

def node_recation(Unit, data, type, total_nodes_num, threshold):
    
    k = data[0]
    v = XX_Sent[k]
    v = np.where(v == 1)[0]
    v = v.tolist()
    #index = raw_S.loc[v].drop("sen_list", axis=1)
    index = raw_S.loc[v]
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
    merged_df = pd.merge(merged_df, X, on='doc_id', how='left')
    merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
    #merged_df[['doc_id', 'sen_id']] = np.sort(merged_df[['doc_id', 'sen_id']], axis=1)
    #merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id']).reset_index(drop=True)
    
    return merged_df, k
    
# In[]
#data = 'ABL_UK'
#from_token = "ABL"
#to_token = "UK"

def edge_recation(Unit, data, type, total_nodes_num, threshold):
    
    from_token = ''
    to_token = ''
    for i, j in zip(coo_df['item1'], coo_df['item2']):
        if data[0] == '{}_{}'.format(i, j):
            from_token = i
            to_token = j
            break
        
    if Unit == "Sentence":
        #XX = XX_Sent
        
        token_df = XX_Sent[[from_token, to_token]].copy()
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] == 2]
        
        index = raw_S.loc[token_df.index.tolist()]
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
        
    else:
        
        #XX = XX_Doc
        token_df = XX_Sent[[from_token, to_token]].copy()
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] >= 1]
        
        index = raw_S.loc[token_df.index.tolist()]
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
        
    return merged_df2, from_token


@app.callback(
    Output('table', 'data'),
    Input('dropdown_choose_SenorDoc', 'value'),
    Input('net', 'selection'),
    Input("total_nodes_num_slider", "value"),
    Input('dropdown-update-layout', 'value'),
    Input('threshold_slide', 'value'),
)
def update_elements(Unit, selection, total_nodes_num, type, threshold):
    global merged_df
    res = []
    
    if len(selection['nodes']) != 0:
        print(selection)
        merged_df, token = node_recation(Unit, selection['nodes'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df['artDate'], merged_df['doc_id'], merged_df['sen_list'], merged_df['artUrl']):
            res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '15%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent:{}'.format(token),
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '50%'},
            {'title': 'url',
            'dataIndex': 'url',
            'key': 'url',
            'width': '15%'}
        ]
    elif len(selection['edges']) != 0:
        print(selection)
        merged_df2, token = edge_recation(Unit, selection['edges'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df2['artDate'], merged_df2['doc_id'], merged_df2['sen_list'], merged_df2['artUrl']):
            res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '15%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent:{}'.format(token),
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '50%'},
            {'title': 'url',
            'dataIndex': 'url',
            'key': 'url',
            'width': '15%'}
        ]
    else:
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '15%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent',
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '50%'},
            {'title': 'url',
            'dataIndex': 'url',
            'key': 'url',
            'width': '15%'}
        ]
        
    table_data['dataSource'] = res

    return table_data

@app.callback(
    Output('textarea-example', 'value'),
    Input('table', 'box_selected_keys')
)
def myfun(box_selected_keys): 
    print([box_selected_keys[0]])
    if box_selected_keys == None:
        return ''
    else: 
        return merged_df['artContent'][box_selected_keys[0]]

app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter