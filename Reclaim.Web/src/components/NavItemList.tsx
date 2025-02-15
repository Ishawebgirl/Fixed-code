import NavItem from './NavItem'

interface INavItemList {
  role: string
}

const NavItemList = ({ role }: INavItemList) => {
  switch (role) {
    case 'Administrator':
      return (
        <ul className="nav flex-column">
          <NavItem label="Dashboard" icon="Home" href="/dashboard"></NavItem>
          <NavItem label="Claims" icon="FolderOpen" href="/claims"></NavItem>
          <NavItem label="Investigators" icon="Clipboard" href="/investigators"></NavItem>
          <NavItem label="Customers" icon="University" href="/customers"></NavItem>
          <NavItem
            label="Administrators"
            icon="ChessKing"
            href="/administrators"></NavItem>
          <NavItem label="Current sign-ins" icon="NetworkWired" href="/signins"></NavItem>
          <NavItem label="Scheduled jobs" icon="PersonDigging" href="/jobs"></NavItem>
        </ul>
      )

    case 'Investigator':
      return (
        <ul className="nav flex-column">
          <NavItem label="Dashboard" icon="Home" href="/dashboard"></NavItem>
          <NavItem label="Claims" icon="FolderOpen" href="/claims"></NavItem>
          <NavItem label="Customers" icon="University" href="/customers"></NavItem>
        </ul>
      )

    case 'Customer':
      return (
        <ul className="nav flex-column">
          <NavItem label="Dashboard" icon="Home" href="/dashboard"></NavItem>
          <NavItem label="Claims" icon="FolderOpen" href="/claims"></NavItem>
          <NavItem label="Investigators" icon="Clipboard" href="/investigators"></NavItem>
        </ul>
      )

    default:
      return <></>
  }
}

export default NavItemList
