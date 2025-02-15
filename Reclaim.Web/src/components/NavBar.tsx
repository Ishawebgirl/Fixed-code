import NavItemList from './NavItemList'

interface INavBar {
  role: string
}

const NavBar = ({ role }: INavBar) => {
  return (
    <nav id="nav-bar" className="nav-bar">
      <div className="position-sticky pt-md-5">
        <div className="logo">
          <div className="logo-img"></div>
        </div>
        <NavItemList role={role}></NavItemList>
      </div>
    </nav>
  )
}

export default NavBar
