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

mcp = FastMCP("Auto Repair Shop - Parts Supplier Server")

# Comprehensive parts inventory mapped to mechanic server error codes
PARTS_INVENTORY = {
    # Parts for P0300 (Misfire) - Referenced in mechanic server
    "12345": {
        "part_number": "12345",
        "description": "Brake Pads - Front Set Premium",
        "brand": "OEM Toyota",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2019", "Toyota Camry 2020"],
        "price": 89.99,
        "stock_quantity": 15,
        "warehouse": "Main Warehouse",
        "category": "Brakes",
        "compatible_error_codes": ["B1234"],
        "installation_time": "1.5 hours"
    },
    "67890": {
        "part_number": "67890", 
        "description": "Oil Filter - Premium OEM",
        "brand": "Genuine Toyota",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2017", "Toyota Camry 2019"],
        "price": 12.50,
        "stock_quantity": 45,
        "warehouse": "Main Warehouse",
        "category": "Engine",
        "compatible_error_codes": ["P0300"],
        "installation_time": "0.5 hours"
    },
    "22222": {
        "part_number": "22222",
        "description": "Spark Plugs Set (4pc) - Iridium",
        "brand": "NGK Iridium",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2019"],
        "price": 48.75,
        "stock_quantity": 8,
        "warehouse": "Secondary Warehouse",
        "category": "Ignition",
        "compatible_error_codes": ["P0300"],
        "installation_time": "1 hour"
    },
    "11111": {
        "part_number": "11111",
        "description": "Engine Air Filter - High Flow",
        "brand": "OEM Toyota",
        "fits": ["Toyota Camry 2018"],
        "price": 24.99,
        "stock_quantity": 0,  # Out of stock for testing
        "warehouse": "Main Warehouse",
        "category": "Engine",
        "compatible_error_codes": ["P0171"],
        "installation_time": "0.25 hours",
        "restock_date": "2025-06-02"
    },
    "33333": {
        "part_number": "33333",
        "description": "Ignition Coil Set (4pc)",
        "brand": "Denso OEM",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2017"],
        "price": 185.60,
        "stock_quantity": 3,
        "warehouse": "Main Warehouse", 
        "category": "Ignition",
        "compatible_error_codes": ["P0300"],
        "installation_time": "2 hours"
    },
    "44444": {
        "part_number": "44444",
        "description": "Fuel Injector Set (4pc)",
        "brand": "Bosch OEM",
        "fits": ["Toyota Camry 2018"],
        "price": 285.00,
        "stock_quantity": 2,
        "warehouse": "Main Warehouse",
        "category": "Fuel System",
        "compatible_error_codes": ["P0300"],
        "installation_time": "3 hours"
    },
    "55555": {
        "part_number": "55555",
        "description": "Mass Air Flow Sensor",
        "brand": "Denso OEM",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2019"],
        "price": 145.99,
        "stock_quantity": 4,
        "warehouse": "Secondary Warehouse",
        "category": "Engine",
        "compatible_error_codes": ["P0171"],
        "installation_time": "0.5 hours"
    },
    "33444": {
        "part_number": "33444",
        "description": "Vacuum Hose Kit",
        "brand": "OEM Toyota",
        "fits": ["Toyota Camry 2018", "Toyota Camry 2017", "Toyota Camry 2019"],
        "price": 35.50,
        "stock_quantity": 12,
        "warehouse": "Main Warehouse",
        "category": "Engine",
        "compatible_error_codes": ["P0171"],
        "installation_time": "1.5 hours"
    },
    "55666": {
        "part_number": "55666",
        "description": "Brake Fluid - DOT 3 (1L)",
        "brand": "Genuine Toyota",
        "fits": ["Toyota Camry 2018", "Honda Accord 2020", "Universal"],
        "price": 15.99,
        "stock_quantity": 25,
        "warehouse": "Main Warehouse",
        "category": "Brakes",
        "compatible_error_codes": ["B1234"],
        "installation_time": "0.25 hours"
    }
}

SUPPLIERS_NETWORK = {
    "local": {
        "name": "Local Auto Parts Direct",
        "delivery_time": "Same day",
        "delivery_cost": 5.00,
        "minimum_order": 25.00
    },
    "regional": {
        "name": "Regional Parts Distribution",
        "delivery_time": "Next day", 
        "delivery_cost": 12.50,
        "minimum_order": 50.00
    },
    "manufacturer": {
        "name": "Toyota Parts Direct",
        "delivery_time": "2-3 days",
        "delivery_cost": 0.00,
        "minimum_order": 100.00
    }
}

@mcp.tool()
def check_part_availability(
    part_number: str,
    vehicle_make: str = "",
    vehicle_model: str = "", 
    vehicle_year: int = None,
    quantity_needed: int = 1,
    urgency: str = "standard"
) -> Dict:
    """
    Check if a specific part is in stock for a vehicle - Primary tool for mechanic interactions.
    :param part_number: the part number to search for
    :param vehicle_make: vehicle manufacturer (e.g., Toyota)
    :param vehicle_model: vehicle model (e.g., Camry) 
    :param vehicle_year: vehicle year (e.g., 2018)
    :param quantity_needed: how many parts are needed
    :param urgency: urgency level (standard, urgent, emergency)
    :return: comprehensive availability information
    """
    
    part_info = PARTS_INVENTORY.get(part_number)
    
    if not part_info:
        return {
            "part_number": part_number,
            "found": False,
            "message": f"❌ Part #{part_number} not found in our catalog",
            "suggestions": [
                "Verify part number with vehicle manual",
                "Contact technical support for alternative parts",
                "Check if part number format is correct"
            ],
            "alternative_search": True
        }
    
    # Check vehicle compatibility
    vehicle_string = f"{vehicle_make} {vehicle_model} {vehicle_year}".strip()
    is_compatible = True
    compatibility_message = "Vehicle compatibility not specified"
    
    if vehicle_make and vehicle_model and vehicle_year:
        is_compatible = any(vehicle_string in fit for fit in part_info["fits"])
        if is_compatible:
            compatibility_message = f"✅ Confirmed compatible with {vehicle_string}"
        else:
            compatibility_message = f"⚠️  May not be compatible with {vehicle_string}. This part fits: {', '.join(part_info['fits'])}"
    
    # Determine availability status
    in_stock = part_info["stock_quantity"] >= quantity_needed
    availability_status = "In Stock"
    
    if not in_stock:
        if part_info["stock_quantity"] > 0:
            availability_status = f"Limited Stock (only {part_info['stock_quantity']} available)"
        else:
            availability_status = "Out of Stock"
    
    # Determine best supplier based on urgency and stock
    recommended_supplier = "local"
    if urgency == "emergency" and in_stock:
        recommended_supplier = "local"
    elif urgency == "urgent":
        recommended_supplier = "regional"
    elif not in_stock:
        recommended_supplier = "manufacturer"
    
    supplier_info = SUPPLIERS_NETWORK[recommended_supplier]
    
    result = {
        "part_number": part_number,
        "found": True,
        "description": part_info["description"],
        "brand": part_info["brand"],
        "price": part_info["price"],
        "stock_quantity": part_info["stock_quantity"],
        "quantity_needed": quantity_needed,
        "in_stock": in_stock,
        "warehouse": part_info["warehouse"],
        "category": part_info["category"],
        "vehicle_compatibility": compatibility_message,
        "is_compatible": is_compatible,
        "availability_status": availability_status,
        "installation_time": part_info.get("installation_time", "Unknown"),
        "compatible_error_codes": part_info.get("compatible_error_codes", []),
        "urgency_level": urgency,
        "recommended_supplier": supplier_info,
        "total_cost": round(part_info["price"] * quantity_needed, 2) if in_stock else None,
        "mechanic_notes": f"This part addresses error codes: {', '.join(part_info.get('compatible_error_codes', ['General']))}"
    }
    
    # Add restock information if out of stock
    if not in_stock and "restock_date" in part_info:
        result["restock_date"] = part_info["restock_date"]
        result["restock_message"] = f"Expected back in stock: {part_info['restock_date']}"
    
    return result

@mcp.tool()
def create_parts_order(
    part_numbers: List[str],
    quantities: List[int],
    work_order_number: str,
    delivery_priority: str = "standard",
    customer_po: str = None
) -> Dict:
    """
    Create an order for parts needed for repair work.
    :param part_numbers: list of part numbers to order
    :param quantities: quantities for each part
    :param work_order_number: associated work order
    :param delivery_priority: standard, urgent, or emergency
    :param customer_po: customer purchase order number
    :return: order confirmation with delivery details
    """
    
    order_number = f"PO{random.randint(100000, 999999)}"
    order_items = []
    subtotal = 0
    
    # Process each part
    for i, part_num in enumerate(part_numbers):
        qty = quantities[i] if i < len(quantities) else 1
        part_info = PARTS_INVENTORY.get(part_num)
        
        if part_info:
            line_total = part_info["price"] * qty
            subtotal += line_total
            
            order_items.append({
                "part_number": part_num,
                "description": part_info["description"],
                "brand": part_info["brand"],
                "quantity": qty,
                "unit_price": part_info["price"],
                "line_total": round(line_total, 2),
                "expected_delivery": calculate_delivery_date(delivery_priority, part_info["warehouse"])
            })
    
    # Calculate totals
    tax_rate = 0.08
    tax_amount = subtotal * tax_rate
    delivery_charge = get_delivery_charge(delivery_priority, subtotal)
    total = subtotal + tax_amount + delivery_charge
    
    return {
        "order_number": order_number,
        "work_order_reference": work_order_number,
        "customer_po": customer_po,
        "order_date": datetime.now().strftime("%Y-%m-%d"),
        "delivery_priority": delivery_priority,
        "items": order_items,
        "subtotal": round(subtotal, 2),
        "tax_amount": round(tax_amount, 2),
        "delivery_charge": round(delivery_charge, 2),
        "total": round(total, 2),
        "estimated_delivery": calculate_delivery_date(delivery_priority),
        "order_status": "Confirmed - Processing",
        "tracking_available": True,
        "special_instructions": get_delivery_instructions(delivery_priority)
    }


def calculate_delivery_date(priority: str, warehouse: str = "Main Warehouse") -> str:
    """Calculate delivery date based on priority and warehouse location."""
    base_days = 1 if warehouse == "Main Warehouse" else 2
    
    if priority == "emergency":
        return datetime.now().strftime("%Y-%m-%d") + " (Same day)"
    elif priority == "urgent":
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        return (datetime.now() + timedelta(days=base_days)).strftime("%Y-%m-%d")

def get_delivery_charge(priority: str, subtotal: float) -> float:
    """Calculate delivery charge based on priority and order value."""
    if subtotal > 100:
        return 0.0  # Free delivery over $100
    elif priority == "emergency":
        return 25.0
    elif priority == "urgent":
        return 15.0
    else:
        return 8.50

def get_delivery_instructions(priority: str) -> str:
    """Get special delivery instructions based on priority."""
    if priority == "emergency":
        return "Rush delivery - Direct to technician bay"
    elif priority == "urgent":
        return "Priority handling - Notify upon arrival"
    else:
        return "Standard delivery to parts counter"

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

    parser = argparse.ArgumentParser(description='Run Auto Repair Shop Parts Supplier MCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    args = parser.parse_args()

    print(f"📦 Starting Auto Repair Shop - Parts Supplier Server on {args.host}:{args.port}")
    print("\n🔧 Available Parts Services:")
    print("- check_part_availability: Check specific parts for mechanic requests")
    print("- create_parts_order: Process parts orders with delivery tracking")
    
    print(f"\n📋 Sample Parts Inventory:")
    print(f"  - Part #12345: Brake Pads (Stock: 15) - Fixes B1234 errors")
    print(f"  - Part #22222: Spark Plugs (Stock: 8) - Fixes P0300 errors") 
    print(f"  - Part #11111: Air Filter (Stock: 0) - Out of stock example")
    
    print(f"\n🔗 Ready to coordinate with Mechanic Server on port 8080")
    print(f"💡 Example: 'Do you have part #12345 in stock for a Toyota Camry 2018?'")

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)