import complete_workflow as cf

def extract_text():
    for i in range(1, 6):
        file_name = "answer_" + str(i)
        print(f"Extracting Text From - {file_name}.wav")

        input_file = './audios/answer_' + str(i) + '.wav'

        output = cf.extract_text(input_file)  

        with open("./transcript/" + file_name + ".txt", "w") as text_file:
            text_file.write(output) 

