"""
Permission management for role-based access control
"""

from typing import List, Dict, Set
from core.models import UserRole, DataSourceType


class PermissionManager:
    """Manages role-based permissions for data sources"""
    
    def __init__(self):
        # Define default permissions for each role
        self.role_permissions = {
            UserRole.ADMIN: {
                DataSourceType.DOCUMENTATION,
                DataSourceType.TICKETS,
                DataSourceType.RUNBOOKS
            },
            UserRole.ENGINEER: {
                DataSourceType.DOCUMENTATION,
                DataSourceType.RUNBOOKS,
                DataSourceType.TICKETS,
            },
            UserRole.SUPPORT: {
                DataSourceType.TICKETS
            },
            UserRole.SALES: {
                DataSourceType.TICKETS
            },
            UserRole.MANAGER: {
                DataSourceType.RUNBOOKS,
                DataSourceType.TICKETS
            },
            UserRole.HR: {
                DataSourceType.DOCUMENTATION,
                DataSourceType.TICKETS
            }
        }
        
        # User-specific restrictions (can be added dynamically)
        self.user_restrictions: Dict[str, Set[DataSourceType]] = {}
    
    def get_accessible_sources(self, role: UserRole) -> List[str]:
        """Get list of accessible data source types for a role"""
        return [source_type.value for source_type in self.role_permissions.get(role, set())]
    
    def get_unauthorized_message(self, role: UserRole, source_type: DataSourceType) -> str:
        """Get standardized unauthorized access message"""
        return f"Access Denied: Role '{role.value}' is not authorized to access '{source_type.value}' data."

    def can_access_source(self, role: UserRole, source_type: DataSourceType) -> bool:
        """Check if a role can access a specific data source type"""
        return source_type in self.role_permissions.get(role, set())
    
    def get_accessible_servers(self, role: UserRole) -> List[str]:
        """Get list of accessible server names for a role"""
        # Map data source types to server names
        server_mapping = {
            DataSourceType.DOCUMENTATION: "documentation",
            DataSourceType.TICKETS: "tickets",
            DataSourceType.RUNBOOKS: "runbooks"
        }
        
        accessible_sources = self.role_permissions.get(role, set())
        return [server_mapping[source] for source in accessible_sources if source in server_mapping]
    
    def add_user_restriction(self, user_id: str, restricted_sources: List[DataSourceType]):
        """Add custom restrictions for a specific user"""
        if user_id not in self.user_restrictions:
            self.user_restrictions[user_id] = set()
        self.user_restrictions[user_id].update(restricted_sources)
    
    def remove_user_restriction(self, user_id: str, restricted_sources: List[DataSourceType]):
        """Remove custom restrictions for a specific user"""
        if user_id in self.user_restrictions:
            self.user_restrictions[user_id].difference_update(restricted_sources)
    
    def can_user_access_source(self, user_id: str, role: UserRole, source_type: DataSourceType) -> bool:
        """Check if a specific user can access a data source (including custom restrictions)"""
        # Check role-based permissions first
        if not self.can_access_source(role, source_type):
            return False
        
        # Check user-specific restrictions
        if user_id in self.user_restrictions:
            return source_type not in self.user_restrictions[user_id]
        
        return True
    
    def get_user_accessible_sources(self, user_id: str, role: UserRole) -> List[DataSourceType]:
        """Get accessible sources for a specific user (including custom restrictions)"""
        accessible_sources = self.role_permissions.get(role, set())
        
        # Apply user restrictions
        if user_id in self.user_restrictions:
            accessible_sources = accessible_sources - self.user_restrictions[user_id]
        
        return list(accessible_sources)
