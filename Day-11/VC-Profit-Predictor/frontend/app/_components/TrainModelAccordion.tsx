import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { HelpCircle } from "lucide-react";

export default function TrainModelAccordion() {
  return (
    <Accordion type="single" collapsible className="w-full">
      <AccordionItem value="item-1">
        <AccordionTrigger className="text-sm">
          <div className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            File Format Requirements
          </div>
        </AccordionTrigger>

        <AccordionContent className="text-xs md:text-sm">
          <p className="text-muted-foreground mb-2">
            Your CSV or JSON file should contain the following columns:
          </p>

          <ul className="list-disc list-inside text-muted-foreground space-y-1 ml-2">
            <li>R&D Spend (numeric)</li>
            <li>Administration (numeric)</li>
            <li>Marketing Spend (numeric)</li>
            <li>State (categorical: New York, California, or Florida)</li>
            <li>Profit (numeric)</li>
          </ul>

          <p className="text-muted-foreground mt-2">Example CSV format:</p>

          <pre className="bg-muted p-2 rounded mt-1 overflow-x-auto text-[10px] md:text-xs">
            {`R&D Spend,Administration,Marketing Spend,State,Profit
142107.3,91391.7,366168.4,Florida,166187.9`}
          </pre>

          <p className="text-muted-foreground mt-2">Example JSON format:</p>

          <pre className="bg-muted p-2 rounded mt-1 overflow-x-auto text-[10px] md:text-xs">
            {`[
  {
    "R&D Spend": 142107.34,
    "Administration": 91391.77,
    "Marketing Spend": 366168.42,
    "State": "Florida",
    "Profit": 166187.94
  }
]`}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}
