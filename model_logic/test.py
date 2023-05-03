import re
def getCombined(easy,cnn):
    prefix=""
    middle=""
    first_numbers=""
    last_numbers=""
    easy_prefix=easy[:2]
    if("ga" in easy):
        easy_prefix="ga"
    if("pa" in easy):
        easy_middle="pa"
    elif("kha" in easy):
        easy_middle="kha"
    else:
        easy_middle="cha"
    if(len(easy)<4):
        return '0'
    if(len(cnn)<4):
        return '0'
    if(len(re.findall(r'\d+(?=[A-Za-z])', easy))>0):
        easy_first_numbers = re.findall(r'\d+(?=[A-Za-z])', easy)[0]
    if(len(easy_first_numbers))>2:
        easy_first_numbers=easy_first_numbers[:2]
    if(len(re.findall(r'[A-Za-z](\d+)', easy))>0):
        easy_last_numbers = re.findall(r'[A-Za-z](\d+)', easy)[-1]
    if(len(easy_last_numbers))>4:
        easy_last_numbers=easy_last_numbers[-4:]
    
    easy_modified=easy_prefix+easy_first_numbers+easy_middle+easy_last_numbers

    cnn_prefix=cnn[:2]
    cnn_middle="pa"
    test1=re.findall(r'[A-Za-z](\d+)', cnn)
    cnn_first_numbers=""
    if(len(test1)>0):
      cnn_first_numbers = re.findall(r'[A-Za-z](\d+)', cnn)[0]
    if(len(cnn_first_numbers))>2:
        cnn_first_numbers=cnn_first_numbers[:2]
    if(len(re.findall(r'[A-Za-z](\d+)', cnn))>0):
        cnn_last_numbers = re.findall(r'[A-Za-z](\d+)', cnn)[-1]
    if(len(cnn_last_numbers))>4:
        cnn_last_numbers=cnn_last_numbers[-4:]
    cnn_modified=cnn_prefix+cnn_first_numbers+cnn_middle+cnn_last_numbers
    prefix="ba"
    first_numbers=cnn_first_numbers
    last_numbers=cnn_last_numbers
    middle=easy_middle
    if("ga" in easy):
        return easy
    if (cnn=='0'):
        return easy
    if ("cha" in easy):
        middle="cha"
    if(len(cnn)<9) and cnn[:-4].isnumeric():
        intermidiate_text=easy[:len(easy) - 4]+cnn[:-4]
        return intermidiate_text
    if(len(first_numbers)<1):
        first_numbers=easy_first_numbers
    if(len(last_numbers)<3):
        last_numbers=easy_last_numbers
    print(cnn_first_numbers)
    return (prefix+first_numbers+middle+last_numbers)

print(getCombined('ba25pa2578','ba19825778'))