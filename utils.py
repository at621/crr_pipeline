import pandas as pd
import re
import logging
import tiktoken
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class DocumentCategorizer:
    def __init__(self, category_1_list, category_2_list, llm, cost_tracker):
        self.llm, self.cost_tracker = llm, cost_tracker
        class Categories(BaseModel):
            category_1: str = Field(description=f"The approach/framework from the list: {category_1_list}")
            category_2: str = Field(description=f"The risk parameter from the list: {category_2_list}")
        self.parser = PydanticOutputParser(pydantic_object=Categories)
        self.prompt = PromptTemplate(
            template="Analyze the text and assign categories.\n{format_instructions}\nText: \"{text}\"",
            input_variables=["text"], partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = self.prompt | self.llm | self.parser
        
    def categorize_text(self, text: str):
        try:
            # Ensure text is a string
            text = str(text) if not isinstance(text, str) else text
            # Limit text length to avoid token limits
            text = text[:2000] if len(text) > 2000 else text
            
            input_tokens, result = len(text) // 4, self.chain.invoke({"text": text})
            output_tokens = len(str(result)) // 4
            self.cost_tracker.add_cost(input_tokens, "llm_input", "setup_categorization")
            self.cost_tracker.add_cost(output_tokens, "llm_output", "setup_categorization")
            return result.category_1, result.category_2
        except Exception as e: 
            print(f"Error categorizing text: {e}")
            return "Uncategorized", "Uncategorized"
            
    def categorize_dataset(self, df: pd.DataFrame):
        df_copy = df.copy()
        
        # Use tqdm.pandas() for better integration
        tqdm.pandas(desc="Categorizing Documents")
        
        # Apply function with progress bar
        results = df_copy.progress_apply(
            lambda row: self.categorize_text(str(row['Text']) if pd.notna(row['Text']) else ""),
            axis=1
        )
        
        # Unpack results
        df_copy['Category_1'] = results.apply(lambda x: x[0])
        df_copy['Category_2'] = results.apply(lambda x: x[1])
        
        return df_copy
        

class CostTracker:
    def __init__(self, config):
        self.config, self.total_cost, self.cost_breakdown = config, 0, {
            "setup_categorization": 0, "setup_embedding": 0, "query_categorization": 0,
            "query_embedding": 0, "query_llm_context": 0
        }
    def _calculate_cost(self, tokens, type):
        if type == "embedding": return (tokens / 1_000_000) * self.config['cost_embedding_per_1m_tokens']
        if type == "llm_input": return (tokens / 1_000_000) * self.config['cost_llm_input_per_1m_tokens']
        if type == "llm_output": return (tokens / 1_000_000) * self.config['cost_llm_output_per_1m_tokens']
        return 0
        
    def add_cost(self, tokens, type, component):
        cost = self._calculate_cost(tokens, type)
        self.total_cost += cost
        if component in self.cost_breakdown: self.cost_breakdown[component] += cost
        return cost
        
    def get_summary(self): return {"total_cost": self.total_cost, "breakdown": self.cost_breakdown}


def print_summary(df):
    """Print a concise summary of the parsed data"""
    print("\n" + "="*60)
    print("DOCUMENT STRUCTURE SUMMARY")
    print("="*60)
    
    # Document hierarchy
    hierarchy_cols = [col for col in df.columns if col not in 
                     ['Article_Number', 'Article_Heading', 'Text', 'Token_Count', 'Ends_With_Dot', 'Text_With_Pagebreaks']]
    
    if hierarchy_cols:
        print("\nDocument hierarchy:")
        for base in ['Part', 'Title', 'Chapter', 'Section', 'Subsection']:
            if base in df.columns:
                count = df[base].nunique()
                if count > 0:
                    has_heading = f"{base}_Heading" in df.columns
                    print(f"  - {base}: {count} unique {'(with headings)' if has_heading else ''}")
    
    # Validation summary
    print(f"\nValidation summary:")
    print(f"  - Total tokens: {df['Token_Count'].sum():,}")
    print(f"  - Average tokens per article: {df['Token_Count'].mean():.0f}")
    print(f"  - Articles ending with period: {df['Ends_With_Dot'].sum()}/{len(df)}")
    
    # Sample articles
    print(f"\nFirst 5 articles:")
    print(df[['Article_Number', 'Article_Heading', 'Token_Count']].head())
    
    # Numbering patterns
    print("\nNumbering patterns found:")
    for pattern, desc in [(r'\(\d+\)', '(1), (2), (3)'), 
                         (r'\([a-z]\)', '(a), (b), (c)')]:
        count = df['Text'].apply(lambda x: bool(re.search(pattern, x))).sum()
        print(f"  - Articles with {desc}: {count}")


def check_article(df, article_number):
    """Check a specific article for content and numbering"""
    article = df[df['Article_Number'] == str(article_number)]
    if article.empty:
        print(f"Article {article_number} not found")
        return
    
    row = article.iloc[0]
    print(f"\nArticle {article_number}: {row['Article_Heading']}")
    print(f"Tokens: {row['Token_Count']}, Ends with dot: {row['Ends_With_Dot']}")
    
    # Check numbering patterns
    text = row['Text']
    patterns = {
        'Main (1), (2)': re.findall(r'\(\d+\)', text),
        'Letters (a), (b)': re.findall(r'\([a-z]\)', text),
        'Roman (i), (ii)': re.findall(r'\([ivxlcdm]+\)', text, re.IGNORECASE)
    }
    
    print("\nNumbering found:")
    for desc, matches in patterns.items():
        if matches:
            print(f"  {desc}: {', '.join(matches[:5])}{' ...' if len(matches) > 5 else ''}")
    
    print(f"\nText preview:")
    print("-" * 60)
    print(text[:500] + "..." if len(text) > 500 else text)