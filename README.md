# question_quality_assessment
Task 3 of Diagnostic Questions competition, which was one of the NeurIPS 2020 Competitions.

## Features

### Correct Rate & Correct Rate Inverse
The correct answer rate per question and the value that inverted it (=the wrong answer rate)
```
result_df = pd.DataFrame()
result_df['Correct_Rate'] = data.groupby('QuestionId').agg(['mean'])['IsCorrect']
result_df['Correct_Rate_Inverse'] = 1-result_df['Correct_Rate']
```

### Subject Score
Skill/concept related score associated with the problem
- Less skill/concept associated with the problem, higher score
- (data/metadata/student_metadata_task_3_4.csv)
```
import ast

question_data = question_metadata.sort_values('QuestionId')
question_data['SubjectList'] = question_data['SubjectId'].apply(lambda x: ast.literal_eval(x))
question_data['SubjectLen'] = question_data['SubjectList'].apply(len)
```

### AnswerValue Entropy
the entropy of the normalized value counts of the **AnswerValue** column grouped by the QuestionId
```
result_df['AnswerValue_Entropy'] = data.groupby('QuestionId')['AnswerValue'].agg(lambda x:multinomial.entropy(1,x.value_counts(normalize=True)).mean())
```

### IsCorrect Entropy
the entropy of the normalized value counts of the **IsCorrect** column grouped by the QuestionId
```
result_df['IsCorrect_Entropy'] = data.groupby('QuestionId')['IsCorrect'].agg(lambda x:multinomial.entropy(1,x.value_counts(normalize=True)).mean())
```

### Question Text Length
Number of characters in the problem
- Use Python's EasyOCR module to recognize characters in an image
- (data/images, image2text.ipynb)
```
import easyocr

reader = easyocr.Reader(['en'])
image2text = []
question_id = []

for idx in range(0, 948):
  result = reader.readtext('git_dir/data/images/'+str(idx)+'.jpg')
  question_text = ''
  for data in result:
    question_text += data[1]
  image2text.append(question_text)
  question_id.append(idx)
```

## Best subset selection
Evaluate all possible combinations to find the best parameter combination of the above features.
```
import itertools

# Calculate the combination of all columns
combinations = []
column_combinations = []
num_columns = len(result_df.columns)

for r in range(1, num_columns+1):
    for cols in itertools.combinations(result_df.columns, r):
        combination_sum = result_df[list(cols)].sum(axis=1)
        combinations.append(combination_sum)
        column_combinations.append(list(cols))  # Save column name combinations

subset_df = pd.concat(combinations, axis=1)
subset_df.columns = ['Combination_{}'.format(i) for i in range(len(combinations))]
  
print(column_combinations[62])
subset_df.head()
```

## Scaling
Scaling processing to improve performance
#### Before
![before_scaling](https://github.com/leehahoon/question_quality_assessment/assets/15906121/4c116b4c-9436-4514-b79e-eed403f6fb3a)

#### After
![after_scaling](https://github.com/leehahoon/question_quality_assessment/assets/15906121/75b5dfe9-a9cc-4b2b-9810-c71d4364635a)


## PCA
Reduce 6 features to 2 dimensions. 
Interesting information inference.
![pca](https://github.com/leehahoon/question_quality_assessment/assets/15906121/c357fe03-f310-472a-928b-9b20fb8bf154)
