import pymupdf
import spacy
import networkx as nx
import matplotlib.pyplot as plt

class LegalRiskAnalyzer:
    def __init__(self):
        # Load the small, efficient English model for Light RAG tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Define our "Light RAG" keywords for retrieval
        self.risk_definitions = {
            "INDEMNITY": ["indemnify", "hold harmless", "liable", "compensation"],
            "TERMINATION": ["terminate", "cancel", "cease", "breach", "notice period"],
            "JURISDICTION": ["governing law", "arbitration", "courts", "jurisdiction"]
        }

        # Initialize the Knowledge Graph
        self.kg = nx.DiGraph()

    def extract_text(self, pdf_path):
        """Reads the PDF and returns clean text."""
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    def build_knowledge_graph(self, text):
        """
        Core Light RAG Logic:
        1. Split text into processing units (sentences).
        2. Detect if unit contains High Risk keywords.
        3. If Risk detected -> Extract Entities (ORG, DATE, MONEY).
        4. Link Entities to the Risk Clause in the Graph.
        """
        doc = self.nlp(text)

        # Iterate over sentences (clauses)
        for sent in doc.sents:
            sent_text = sent.text.strip()

            # Check which risk category this sentence belongs to
            detected_risk = None
            for category, keywords in self.risk_definitions.items():
                if any(keyword in sent_text.lower() for keyword in keywords):
                    detected_risk = category
                    break

            # If a risk is found, process it into the graph
            if detected_risk:
                # Create a node for the specific clause (truncating text for ID)
                clause_id = f"{detected_risk}_CLAUSE_{hash(sent_text) % 1000}"
                self.kg.add_node(clause_id, type="Clause", text=sent_text, category=detected_risk)

                # Extract entities involved in this risk (The "Who" and "When")
                for ent in sent.ents:
                    # We only care about specific entities for contracts
                    if ent.label_ in ["ORG", "PERSON", "DATE", "MONEY"]:
                        entity_id = ent.text
                        self.kg.add_node(entity_id, type="Entity", label=ent.label_)

                        # Link Entity -> Risk Clause
                        # Meaning: This Entity is involved in this Risk
                        self.kg.add_edge(entity_id, clause_id, relation="associated_with")

    def analyze_risks(self):
        """Retrieves and prints high-risk clusters."""
        print("\n--- HIGH RISK ANALYSIS REPORT ---")

        # Find all Clause nodes
        clause_nodes = [n for n, attr in self.kg.nodes(data=True) if attr.get('type') == 'Clause']

        if not clause_nodes:
            print("No high-risk clauses found.")
            return

        for node in clause_nodes:
            data = self.kg.nodes[node]
            print("============================================================")
            print("============================================================")
            print("============================================================")
            print(f"\n⚠️  Risk Type: {data['category']}")
            print(f"📝 Clause: \"{data['text']}\"")

            # Find connected entities (Predecessors in the directed graph)
            involved_entities = list(self.kg.predecessors(node))
            if involved_entities:
                print(f"🔗 Involved Entities: {', '.join(involved_entities)}")

    def visualize_graph(self):
        """Visualizes the Knowledge Graph."""
        if self.kg.number_of_nodes() == 0:
            print("Graph is empty.")
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.kg, seed=42)

        # Color coding
        color_map = []
        for node in self.kg:
            if self.kg.nodes[node].get('type') == 'Clause':
                color_map.append('salmon')  # Red for risk
            else:
                color_map.append('skyblue')  # Blue for entities

        nx.draw(self.kg, pos, with_labels=True, node_color=color_map,
                node_size=2500, font_size=9, font_weight="bold", edge_color="gray")
        plt.title("Contract Knowledge Graph: Entities vs. Risks")
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup
    # pdf_file = "data_in\\zomato.pdf"
    # pdf_file = "data_in\\lenscart.pdf"
    pdf_file = "data_in\\flipcart.pdf"

    # 2. Run Analysis
    analyzer = LegalRiskAnalyzer()

    print(f"Processing {pdf_file}...")
    raw_text = analyzer.extract_text(pdf_file)
    analyzer.build_knowledge_graph(raw_text)

    # 3. Output Results
    # HIGH RISK ANALYSIS REPORT
    analyzer.analyze_risks()

    # 4. Show Graph
    print("Generating Graph Visualization...")
    analyzer.visualize_graph()

    print("The End")

