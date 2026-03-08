"""
ClinicalKnowledgeEngine: Combines deterministic thresholds and knowledge graph
to generate patient-specific insights and summaries.
"""

from __future__ import annotations
from typing import List, Dict, Any, Union
import networkx as nx


class ClinicalKnowledgeEngine:
    """
    Engine to generate clinical insights and summaries using:
    1. Deterministic thresholds for labs and vitals
    2. Relational knowledge graph for disease-complication relationships
    """

    def __init__(self) -> None:
        # -----------------------------------------------------------------
        # Thresholds for disease management
        # -----------------------------------------------------------------
        self.thresholds: Dict[str, Dict[str, Union[int, float]]] = {
            "Type 2 Diabetes": {"HbA1c": 7.0, "Fasting_Glucose": 126},
            "Hypertension": {"BP_Systolic": 140, "BP_Diastolic": 90},
            "Lipids": {"LDL": 130, "HDL": 45},
            "Kidney": {"eGFR": 60},
        }

        # -----------------------------------------------------------------
        # Knowledge graph: disease → complication → monitored lab
        # -----------------------------------------------------------------
        self.G = nx.DiGraph()

        # Disease → Complication relationships
        self.G.add_edge(
            "Type 2 Diabetes", "Chronic Kidney Disease", relation="is_risk_factor_for"
        )
        self.G.add_edge(
            "Hypertension", "Chronic Kidney Disease", relation="is_risk_factor_for"
        )
        self.G.add_edge(
            "Type 2 Diabetes", "Cardiovascular Disease", relation="is_risk_factor_for"
        )
        self.G.add_edge(
            "Hypertension", "Cardiovascular Disease", relation="is_risk_factor_for"
        )

        # Complication → Lab relationships
        self.G.add_edge("Chronic Kidney Disease", "eGFR", relation="monitored_by")
        self.G.add_edge("Cardiovascular Disease", "LDL", relation="monitored_by")
        self.G.add_edge("Type 2 Diabetes", "HbA1c", relation="monitored_by")

    # -----------------------------------------------------------------
    # Graph Insights
    # -----------------------------------------------------------------

    def get_graph_insights(self, diagnosis_str: str) -> List[str]:
        """
        Traverse the knowledge graph to find complications and monitoring labs
        for one or more diagnoses.

        Parameters
        ----------
        diagnosis_str : str
            Comma-separated list of patient diagnoses.

        Returns
        -------
        List[str]
            Unique graph-derived insights.
        """
        insights: List[str] = []
        diagnoses = [d.strip() for d in diagnosis_str.split(",")]

        for d in diagnoses:
            if self.G.has_node(d):
                for comp in self.G.successors(d):
                    rel = self.G[d][comp]["relation"].replace("_", " ")
                    insights.append(f"Graph Warning: {d} {rel} {comp}.")
        return list(set(insights))  # Deduplicate

    # -----------------------------------------------------------------
    # Generate KG + Lab Summary
    # -----------------------------------------------------------------

    def generate_kg_summary(self, row: Dict[str, Any]) -> str:
        """
        Generate a patient summary combining demographics, knowledge graph insights,
        recent labs, vitals, lifestyle, medications, and symptoms.

        Parameters
        ----------
        row : Dict[str, Any]
            Patient record containing keys like 'Name', 'Age', 'Sex', 'Diagnosis',
            'HbA1c', 'BP_Systolic', 'BP_Diastolic', 'eGFR', 'Exercise_min_per_day',
            'Diet', 'Medications', 'Symptoms'.

        Returns
        -------
        str
            Structured textual summary.
        """
        summary = f" {row['Age']}y {row['Sex']}, diagnosed with {row['Diagnosis']}.\n"

        # Layer 1: Relational Graph Insights
        graph_insights = self.get_graph_insights(row["Diagnosis"])
        if graph_insights:
            summary += "Knowledge Graph Analysis:\n"
            for insight in graph_insights:
                summary += f" • {insight}\n"

        summary += "\nRecent labs and vitals:  \n"

        # Layer 2: Threshold Checks
        # HbA1c
        t_hba1c = self.thresholds["Type 2 Diabetes"]["HbA1c"]
        status_hba1c = "Above target → Uncontrolled" if row["HbA1c"] > t_hba1c else "Controlled"
        summary += f" - HbA1c {row['HbA1c']}% ({status_hba1c})\n"

        # BP
        t_sys = self.thresholds["Hypertension"]["BP_Systolic"]
        t_dia = self.thresholds["Hypertension"]["BP_Diastolic"]
        if row["BP_Systolic"] > t_sys or row["BP_Diastolic"] > t_dia:
            summary += (
                f" - BP {row['BP_Systolic']}/{row['BP_Diastolic']} mmHg "
                "(Above target → Uncontrolled Hypertension)\n"
            )
        else:
            summary += f" - BP {row['BP_Systolic']}/{row['BP_Diastolic']} mmHg (Controlled)\n"

        # eGFR
        t_egfr = self.thresholds["Kidney"]["eGFR"]
        status_egfr = "Reduced kidney function" if row["eGFR"] < t_egfr else "Normal"
        summary += f" - eGFR {row['eGFR']} mL/min/1.73m² ({status_egfr})\n"

        # Lifestyle & Diet
        summary += f"\n**Exercise:** {row.get('Exercise_min_per_day', 0)} min/day  \n **Diet:** {row.get('Diet', 'Unknown')}  \n"

        # Medications
        meds = row.get("Medications", "None")
        med_str = ", ".join(meds) if isinstance(meds, list) else meds
        summary += f"**Current medications:** {med_str}  \n"

        # Symptoms
        symp = row.get("Symptoms", "None")
        symp_str = ", ".join(symp) if isinstance(symp, list) else symp
        summary += f"**Symptoms**: {symp_str}\n"

        return summary

    # -----------------------------------------------------------------
    # Extract Critical Risks
    # -----------------------------------------------------------------

    def extract_critical_risks(self, row: Dict[str, Any]) -> List[str]:
        """
        Return critical lab/vital alerts for UI or warning boxes.

        Parameters
        ----------
        row : Dict[str, Any]
            Patient record.

        Returns
        -------
        List[str]
            List of alert strings.
        """
        critical_risks: List[str] = []

        if row.get("HbA1c", 0) > self.thresholds["Type 2 Diabetes"]["HbA1c"]:
            critical_risks.append(f"🚨 CRITICAL: HbA1c {row['HbA1c']}% (Uncontrolled Diabetes)")

        # Blood Pressure (combined)
        bp_sys = row.get("BP_Systolic", 0)
        bp_dia = row.get("BP_Diastolic", 0)
        if bp_sys > self.thresholds["Hypertension"]["BP_Systolic"] or \
        bp_dia > self.thresholds["Hypertension"]["BP_Diastolic"]:
            critical_risks.append(
                f"🚨 CRITICAL: BP {bp_sys}/{bp_dia} mmHg (Stage 2 Hypertension Risk)"
            )

        if row.get("eGFR", 100) < self.thresholds["Kidney"]["eGFR"]:
            critical_risks.append(
                f"⚠️ RISK: eGFR {row['eGFR']} (Chronic Kidney Disease Warning)"
            )


        return critical_risks