import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route, Mount
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

mcp = FastMCP("Auto Repair Shop - Mechanic Server")

# Vehicle database for realistic scenarios
VEHICLE_DATABASE = {
    "XYZ123": {
        "vehicle_id": "XYZ123",
        "make": "Toyota",
        "model": "Camry",
        "year": 2018,
        "vin": "1234567890ABCDEF0",
        "mileage": 85000,
        "owner": "John Smith",
        "last_service": "2024-03-15"
    },
    "ABC456": {
        "vehicle_id": "ABC456", 
        "make": "Honda",
        "model": "Accord",
        "year": 2020,
        "vin": "ABCDEF1234567890G",
        "mileage": 45000,
        "owner": "Sarah Johnson",
        "last_service": "2024-05-10"
    }
}

# Error codes and their meanings
ERROR_CODES = {
    "P0300": {
        "code": "P0300",
        "description": "Random/Multiple Cylinder Misfire Detected",
        "severity": "Medium",
        "symptoms": ["Engine rattling", "Poor acceleration", "Rough idle"],
        "common_causes": ["Faulty spark plugs", "Ignition coils", "Fuel injectors", "Vacuum leaks"],
        "required_parts": ["12345", "67890", "22222"]  # References to supplier parts
    },
    "P0171": {
        "code": "P0171",
        "description": "System Too Lean (Bank 1)",
        "severity": "Medium",
        "symptoms": ["Poor fuel economy", "Engine hesitation", "Check engine light"],
        "common_causes": ["Dirty air filter", "Vacuum leak", "Faulty MAF sensor"],
        "required_parts": ["11111", "33444"]
    },
    "B1234": {
        "code": "B1234", 
        "description": "Brake System Warning",
        "severity": "High",
        "symptoms": ["Brake pedal feels soft", "Grinding noise", "Brake warning light"],
        "common_causes": ["Worn brake pads", "Low brake fluid", "Brake line leak"],
        "required_parts": ["12345", "55666"]
    }
}

REPAIR_PROCEDURES = {
    "P0300_Toyota_Camry": {
        "procedure_id": "P0300_Toyota_Camry",
        "title": "Toyota Camry P0300 Misfire Repair",
        "steps": [
            "1. Connect diagnostic scanner and verify error code P0300",
            "2. Perform visual inspection of ignition system components",
            "3. Remove and inspect spark plugs for wear or fouling",
            "4. Test ignition coils with multimeter (resistance should be 12-15 ohms)",
            "5. Check fuel injector operation and spray pattern",
            "6. Inspect vacuum lines for cracks or disconnections",
            "7. Replace faulty components as needed",
            "8. Clear codes and perform test drive"
        ],
        "estimated_time": "2-4 hours",
        "difficulty": "Medium",
        "required_tools": ["Diagnostic Scanner", "Multimeter", "Socket Set", "Platform Lift"],
        "safety_notes": ["Ensure engine is cool before starting work", "Disconnect battery when working on ignition system"]
    }
}

@mcp.tool()
def scan_vehicle_for_error_codes(vehicle_id: str) -> Dict:
    """
    Scan vehicle diagnostic system for error codes.
    :param vehicle_id: unique vehicle identifier
    :return: diagnostic scan results with error codes
    """
    
    vehicle = VEHICLE_DATABASE.get(vehicle_id)
    if not vehicle:
        return {
            "error": f"Vehicle {vehicle_id} not found in system",
            "vehicle_id": vehicle_id
        }
    
    # Simulate finding error codes based on common issues
    possible_codes = ["P0300", "P0171", "B1234"]
    found_codes = random.sample(possible_codes, random.randint(1, 2))
    
    scan_results = {
        "vehicle_id": vehicle_id,
        "vehicle_info": vehicle,
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scanner_type": "OBD-II Professional Diagnostic Scanner",
        "error_codes_found": len(found_codes),
        "codes": []
    }
    
    for code in found_codes:
        code_info = ERROR_CODES.get(code, {})
        scan_results["codes"].append({
            "code": code,
            "description": code_info.get("description", "Unknown error"),
            "severity": code_info.get("severity", "Unknown"),
            "status": "Active",
            "freeze_frame_data": f"RPM: {random.randint(600, 800)}, Load: {random.randint(10, 30)}%"
        })
    
    return scan_results

@mcp.tool()
def get_repair_procedure(
    error_code: str,
    vehicle_make: str,
    vehicle_model: str,
    vehicle_year: int = None
) -> Dict:
    """
    Get detailed repair procedure for specific error code and vehicle.
    :param error_code: diagnostic trouble code (e.g., P0300)
    :param vehicle_make: vehicle manufacturer
    :param vehicle_model: vehicle model
    :param vehicle_year: vehicle year
    :return: detailed repair procedure
    """
    
    # Create procedure key
    procedure_key = f"{error_code}_{vehicle_make}_{vehicle_model}"
    procedure = REPAIR_PROCEDURES.get(procedure_key)
    
    if not procedure:
        # Generate generic procedure
        error_info = ERROR_CODES.get(error_code, {})
        procedure = {
            "procedure_id": f"GENERIC_{error_code}",
            "title": f"Generic Repair for {error_code}",
            "error_description": error_info.get("description", "Unknown error code"),
            "steps": [
                "1. Verify error code with diagnostic scanner",
                "2. Perform visual inspection of related components", 
                "3. Test components according to manufacturer specifications",
                "4. Replace faulty parts as identified",
                "5. Clear codes and verify repair"
            ],
            "estimated_time": "1-3 hours",
            "difficulty": "Medium",
            "note": "Generic procedure - consult specific vehicle manual for detailed steps"
        }
    
    # Add parts information from error code data
    error_info = ERROR_CODES.get(error_code, {})
    procedure["required_parts"] = error_info.get("required_parts", [])
    procedure["common_causes"] = error_info.get("common_causes", [])
    procedure["symptoms"] = error_info.get("symptoms", [])
    
    return {
        "vehicle": f"{vehicle_year} {vehicle_make} {vehicle_model}",
        "error_code": error_code,
        "procedure": procedure,
        "parts_needed_check": f"Need to verify availability of parts: {', '.join(procedure['required_parts'])}" if procedure.get('required_parts') else "No specific parts identified yet"
    }

@mcp.tool()
def generate_work_order(
    vehicle_id: str,
    customer_complaint: str,
    diagnostic_findings: Dict,
    recommended_repairs: List[str],
    parts_needed: List[str] = None
) -> Dict:
    """
    Generate comprehensive work order based on diagnosis.
    :param vehicle_id: vehicle identifier
    :param customer_complaint: original customer complaint
    :param diagnostic_findings: results from diagnostic tools
    :param recommended_repairs: list of recommended repair procedures
    :param parts_needed: list of part numbers needed
    :return: complete work order
    """
    
    vehicle = VEHICLE_DATABASE.get(vehicle_id, {})
    work_order_number = f"WO{random.randint(100000, 999999)}"
    
    return {
        "work_order_number": work_order_number,
        "date_created": datetime.now().strftime("%Y-%m-%d"),
        "vehicle_info": vehicle,
        "customer_complaint": customer_complaint,
        "diagnostic_summary": diagnostic_findings,
        "recommended_repairs": recommended_repairs,
        "parts_needed": parts_needed or [],
        "estimated_labor_hours": random.randint(2, 8),
        "labor_rate": "$120/hour",
        "priority": "Medium",
        "assigned_technician": "Master Technician AI",
        "status": "Awaiting parts and customer approval",
        "next_steps": [
            "Verify parts availability with supplier",
            "Get customer approval for repairs",
            "Schedule repair appointment",
            "Order required parts"
        ],
        "notes": "Diagnosis completed. Awaiting parts check and customer authorization to proceed."
    }

@mcp.tool()
def customer_communication_log(
    vehicle_id: str,
    interaction_type: str,
    message: str,
    response: str = None
) -> Dict:
    """
    Log customer communications and diagnostic questions.
    :param vehicle_id: vehicle identifier
    :param interaction_type: type of interaction (diagnostic_question, update, approval_request)
    :param message: message sent to customer
    :param response: customer response if received
    :return: communication log entry
    """
    
    log_entry = {
        "log_id": f"COMM{random.randint(10000, 99999)}",
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "interaction_type": interaction_type,
        "technician_message": message,
        "customer_response": response,
        "status": "Response received" if response else "Awaiting response",
        "follow_up_needed": response is None
    }
    
    return {
        "communication_logged": True,
        "log_entry": log_entry,
        "total_interactions": random.randint(3, 8),
        "next_action": "Wait for customer response" if not response else "Proceed with diagnosis"
    }

def create_starlette_app(
        mcp_server: Server,
        *,
        debug: bool = False,
) -> Starlette:
    """
    Create a Starlette application that can serve the provided mcp server with SSE.
    :param mcp_server: the mcp server to serve
    :param debug: whether to enable debug mode
    :return: a Starlette application
    """

    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    import argparse

    parser = argparse.ArgumentParser(description='Run Auto Repair Shop Mechanic MCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    print(f"🔧 Starting Auto Repair Shop - Mechanic Server on {args.host}:{args.port}")
    print("\n🛠️  Available Diagnostic Tools:")
    print("- scan_vehicle_for_error_codes: OBD-II diagnostic scanning")
    print("- get_repair_procedure: Access repair manual database") 
    print("- raise_platform: Operate hydraulic lift system")
    print("- use_multimeter: Electrical component testing")
    print("- generate_work_order: Create comprehensive work orders")
    print("- customer_communication_log: Track customer interactions")
    
    print(f"\n🚗 Sample Vehicles in System:")
    for vid, info in VEHICLE_DATABASE.items():
        print(f"  - {vid}: {info['year']} {info['make']} {info['model']} ({info['owner']})")
    
    print(f"\n📋 Ready to handle diagnostic workflows and coordinate with Parts Supplier on port 8081")

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)