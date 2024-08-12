import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

def plot_box_result(args):
    model_name = args.test_model_name.split('/')[-1]
    with open(args.test_result_save_dir + f"result_{model_name}.json", 'r') as json_file:  
        data = json.load(json_file)
    
    depth_values = [entry["depth"] for entry in data]  
    context_lengths = [entry["token_lengths"] for entry in data]  
    results = [entry["result"] for entry in data]  
    results_percentage = [result * 100 for result in results]  
    df = pd.DataFrame({'Context Length': context_lengths, 'Depth Values': depth_values, 'Result': results_percentage})  

    average_results = df.groupby('Context Length')['Result'].mean()  
    result_list = average_results.tolist()  
    plt.figure(figsize=(5, 6))  
    unique_context_lengths = list(dict.fromkeys(context_lengths))  
    x_labels = [str(i//1000)+"K" for i in unique_context_lengths]
    x = [i for i in range(len(x_labels))]
    
    line, = plt.plot(x, result_list,marker = "^",markersize=10,linewidth=2.5,zorder =3)  
    plt.xlabel('Token Length')  
    plt.ylabel('Average Acc(%)')  
    
    plt.ylim(0,110)
    plt.xlim(-0.2,len(x)-0.8)
    plt.xticks(x,x_labels, rotation=45)  
    plt.axvspan(-0.2, len(x)-0.8,facecolor='gray', alpha=0.3)  
   
    plt.grid(True,color = "white")
    plt.legend([line], [args.test_model_name], loc='upper right')  
    plt.title(f'{args.test_model_name} box plot')  
    plt.savefig(args.test_result_save_dir + f"{model_name}_box_plot.png", dpi=300)

def plot_line_result(args):
    model_name = args.test_model_name.split('/')[-1]
    with open(args.test_result_save_dir + f"result_{model_name}.json", 'r') as json_file:  
        data = json.load(json_file)
    
    depth_values = [entry["depth"] for entry in data]  
    context_lengths = [entry["token_lengths"] for entry in data]  
    results = [entry["result"] for entry in data]  
    results_percentage = [result * 100 for result in results]  
    df = pd.DataFrame({'Context Length': context_lengths, 'Depth Values': depth_values, 'Result': results_percentage})  
    
    pivot_df = df.pivot_table(index='Depth Values', columns='Context Length', values='Result', aggfunc='mean') 
    plt.figure(figsize=(10, 8))  
    ax = sns.heatmap(pivot_df, cmap='RdYlGn', annot=True, fmt=".0f")  
    for t in ax.texts: t.set_text(t.get_text() + "%")
    plt.xlabel('Tokens Length')  
    plt.ylabel('Depth (%)')  
    plt.title(f'{args.test_model_name} Results')  
    
    plt.savefig(args.test_result_save_dir + f'{model_name}_line_plot.png')


def visualize(args):
    plot_line_result(args)
    plot_box_result(args)