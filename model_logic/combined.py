import nepali_roman as nr

def postProcess(recognized):
    english_converted=""
    prefix=""
    middle=""
    first_numbers=""
    last_numbers=""
    processed_plate=""
    first_numbers_count=0
    last_numbers_count=0
    if(nr.is_devanagari(recognized)):
        english_converted=nr.romanize_text(recognized)
        print(english_converted)
    if(english_converted[:2]!='ga' and english_converted[:2]!='ko' and english_converted[:2]!='na'):
        prefix='ba'
    else:
        prefix=english_converted[:2]
    if("pa" in english_converted):
        middle="pa"
    elif("kha" in english_converted):
        middle="kha"
    else:
        middle="cha"
    for x in english_converted:
        if x.isnumeric():
            first_numbers+=x
            first_numbers_count+=1
        if(first_numbers_count==2):
            break
    for x in reversed(english_converted):
        if x.isnumeric():
            last_numbers+=x
            last_numbers_count+=1
        if(last_numbers_count==4):
            break
    last_numbers= ''.join(reversed(last_numbers))
    processed_plate=prefix+first_numbers+middle+last_numbers
    return processed_plate


def getCombined(easy,cnn):
    print(len(cnn))
    if("ga" in easy):
        return easy
    elif (cnn=='0'):
        return easy
    elif(len(cnn)<9) and cnn[:-4].isnumeric():
        intermidiate_text=easy[:len(easy) - 4]+cnn[:-4]
        return intermidiate_text
    else:
        return cnn
        
        
print(getCombined("ba86pa5888","ba86pa5984"))
