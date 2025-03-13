import uuid
import json
import numpy as np
from typing import Dict, Any, Optional, List


class ModelGene:
    """
    Represents a single gene in the model's DNA.

    Genes define specific characteristics or capabilities of a model.
    """

    def __init__(self, name: str, value: float, mutable: bool = True, mutation_rate: float = 0.1):
        self.name = name
        self.value = value
        self.mutable = mutable
        self.mutation_rate = mutation_rate

    def mutate(self, mutation_strength: float = 0.2) -> None:
        """Apply random mutation to the gene value if it's mutable."""
        if not self.mutable or np.random.random() > self.mutation_rate:
            return

        # Apply random mutation with controlled strength
        mutation = np.random.normal(0, mutation_strength)
        self.value = max(0.0, min(1.0, self.value + mutation))

    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "mutable": self.mutable,
            "mutation_rate": self.mutation_rate
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelGene':
        """Create gene from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            mutable=data["mutable"],
            mutation_rate=data["mutation_rate"]
        )


class ModelDNA:
    """
    Represents the genetic profile of an AI model.

    The DNA defines the model's characteristics, capabilities, and parameters
    that can evolve over time through the genetic algorithm.
    """

    def __init__(self, model_id: Optional[str] = None, **kwargs):
        self.model_id = model_id or str(uuid.uuid4())

        # Initialize base genes
        self.genes = {
            # Accuracy and performance genes
            "precision": ModelGene("precision", kwargs.get("precision", 0.5)),
            "recall": ModelGene("recall", kwargs.get("recall", 0.5)),
            "latency": ModelGene("latency", kwargs.get("latency", 0.5)),

            # Specialization genes
            "numeric_reasoning": ModelGene("numeric_reasoning", kwargs.get("numeric_reasoning", 0.5)),
            "text_understanding": ModelGene("text_understanding", kwargs.get("text_understanding", 0.5)),
            "creative_thinking": ModelGene("creative_thinking", kwargs.get("creative_thinking", 0.5)),

            # Reliability genes
            "consistency": ModelGene("consistency", kwargs.get("consistency", 0.5)),
            "uncertainty_awareness": ModelGene("uncertainty_awareness", kwargs.get("uncertainty_awareness", 0.5)),

            # Bias reduction genes
            "bias_detection": ModelGene("bias_detection", kwargs.get("bias_detection", 0.5)),
            "fairness": ModelGene("fairness", kwargs.get("fairness", 0.5)),
        }

        # Custom genes (domain-specific)
        for key, value in kwargs.items():
            if key not in self.genes and not key.startswith("_"):
                self.genes[key] = ModelGene(key, value)

        # Track ancestry and generation
        self.parent_ids = kwargs.get("parent_ids", [])  # type: List[str]
        self.generation = kwargs.get("generation", 0)
        self.fitness_history = kwargs.get("fitness_history", [])

    def mutate(self, mutation_rate: float = 0.2, mutation_strength: float = 0.2) -> None:
        """Apply random mutations to genes based on mutation rate."""
        for gene in self.genes.values():
            if np.random.random() < mutation_rate:
                gene.mutate(mutation_strength)

    def get_gene_value(self, gene_name: str) -> float:
        """Get the value of a specific gene."""
        if gene_name in self.genes:
            return self.genes[gene_name].value
        return 0.0

    def get_genetic_distance(self, other_dna: 'ModelDNA') -> float:
        """Calculate genetic distance between two DNA profiles."""
        common_genes = set(self.genes.keys()).intersection(other_dna.genes.keys())
        if not common_genes:
            return 1.0  # Maximum distance if no common genes

        total_distance = 0.0
        for gene_name in common_genes:
            value1 = self.genes[gene_name].value
            value2 = other_dna.genes[gene_name].value
            total_distance += abs(value1 - value2)

        return total_distance / len(common_genes)

    def add_fitness_score(self, score: float) -> None:
        """Record a fitness score in the DNA's history."""
        self.fitness_history.append(score)

    def get_average_fitness(self) -> float:
        """Calculate the average fitness from the history."""
        if not self.fitness_history:
            return 0.0
        return sum(self.fitness_history) / len(self.fitness_history)

    def to_dict(self) -> Dict[str, Any]:
        """Convert DNA to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "genes": {name: gene.to_dict() for name, gene in self.genes.items()},
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "fitness_history": self.fitness_history
        }

    def to_json(self) -> str:
        """Convert DNA to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelDNA':
        """Create DNA from dictionary."""
        model_dna = cls(model_id=data["model_id"])
        model_dna.genes = {
            name: ModelGene.from_dict(gene_data)
            for name, gene_data in data["genes"].items()
        }
        model_dna.parent_ids = data["parent_ids"]
        model_dna.generation = data["generation"]
        model_dna.fitness_history = data["fitness_history"]
        return model_dna

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelDNA':
        """Create DNA from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def crossover(cls, dna1: 'ModelDNA', dna2: 'ModelDNA') -> 'ModelDNA':
        """
        Create a new DNA by crossover of two parent DNAs.
        Implements single-point crossover on the gene set.
        """
        # Collect all unique gene names from both parents
        all_genes = sorted(list(set(list(dna1.genes.keys()) + list(dna2.genes.keys()))))

        # Randomly select crossover point
        if len(all_genes) > 1:
            crossover_point = np.random.randint(1, len(all_genes))
        else:
            crossover_point = 0

        # Create gene values for offspring
        offspring_values = {}

        # Take genes from first parent up to crossover point
        for i in range(crossover_point):
            gene_name = all_genes[i]
            if gene_name in dna1.genes:
                offspring_values[gene_name] = dna1.genes[gene_name].value
            elif gene_name in dna2.genes:
                offspring_values[gene_name] = dna2.genes[gene_name].value

        # Take genes from second parent after crossover point
        for i in range(crossover_point, len(all_genes)):
            gene_name = all_genes[i]
            if gene_name in dna2.genes:
                offspring_values[gene_name] = dna2.genes[gene_name].value
            elif gene_name in dna1.genes:
                offspring_values[gene_name] = dna1.genes[gene_name].value

        # Create offspring with new generation number and parent IDs
        parent_ids: List[str] = [dna1.model_id, dna2.model_id]
        generation = max(dna1.generation, dna2.generation) + 1
        
        # Separate gene values from metadata
        return cls(
            model_id=None,  # Explicitly pass None for model_id
            parent_ids=parent_ids,
            generation=generation,
            **offspring_values
        )

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the DNA's key characteristics."""
        # Calculate average values for different categories
        performance_genes = ["precision", "recall", "latency"]
        specialization_genes = ["numeric_reasoning", "text_understanding", "creative_thinking"]
        reliability_genes = ["consistency", "uncertainty_awareness"]
        bias_genes = ["bias_detection", "fairness"]

        def avg_category(category_genes):
            values = [self.get_gene_value(g) for g in category_genes if g in self.genes]
            return sum(values) / len(values) if values else 0

        return {
            "model_id": self.model_id,
            "generation": self.generation,
            "ancestry_depth": len(self.parent_ids),
            "performance_score": avg_category(performance_genes),
            "specialization_score": avg_category(specialization_genes),
            "reliability_score": avg_category(reliability_genes),
            "bias_reduction_score": avg_category(bias_genes),
            "average_fitness": self.get_average_fitness()
        }
